from common.Boilerplate import *
from psycopg2.extras import execute_batch
from datetime import datetime, timedelta
import time

def insert_options_tick_data(conn, tick_data, instrument_info):
    """Insert options tick data into database"""
    try:
        with conn.cursor() as cursor:
            # Insert core tick data
            tick_sql = """
                INSERT INTO my_schema.options_ticks 
                (instrument_token, timestamp, last_trade_time, last_price, last_quantity, buy_quantity, sell_quantity, volume,
                 average_price, oi, oi_day_high, oi_day_low, net_change, lower_circuit_limit, upper_circuit_limit,
                 strike_price, option_type, expiry, tradingsymbol)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """

            tick_values = (
                tick_data['instrument_token'],
                tick_data.get('timestamp') or datetime.now(),
                tick_data.get('last_trade_time'),
                tick_data.get('last_price'),
                tick_data.get('last_quantity'),
                tick_data.get('buy_quantity'),
                tick_data.get('sell_quantity'),
                tick_data.get('volume'),
                tick_data.get('average_price'),
                tick_data.get('oi'),
                tick_data.get('oi_day_high'),
                tick_data.get('oi_day_low'),
                tick_data.get('net_change'),
                tick_data.get('lower_circuit_limit'),
                tick_data.get('upper_circuit_limit'),
                instrument_info.get('strike'),
                instrument_info.get('instrument_type'),  # CE or PE
                instrument_info.get('expiry'),
                instrument_info.get('tradingsymbol', '')
            )
            
            cursor.execute(tick_sql, tick_values)
            result = cursor.fetchone()
            if not result:
                raise Exception("INSERT did not return an id")
            tick_id = result[0]

            # Insert OHLC data
            ohlc = tick_data.get('ohlc')
            if ohlc:
                ohlc_sql = """
                    INSERT INTO my_schema.options_tick_ohlc (tick_id, open, high, low, close)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (tick_id) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close
                """
                cursor.execute(ohlc_sql, (tick_id, ohlc.get('open'), ohlc.get('high'), ohlc.get('low'), ohlc.get('close')))

            # Insert Depth data
            depth = tick_data.get('depth')
            if depth:
                # Delete existing depth data for this tick_id
                cursor.execute("DELETE FROM my_schema.options_tick_depth WHERE tick_id = %s", (tick_id,))
                
                depth_sql = """
                    INSERT INTO my_schema.options_tick_depth (tick_id, side, price, quantity, orders)
                    VALUES (%s, %s, %s, %s, %s)
                """
                depth_rows = []
                for side in ['buy', 'sell']:
                    for order in depth.get(side, []):
                        depth_rows.append((tick_id, side, order.get('price'), order.get('quantity'), order.get('orders')))
                if depth_rows:
                    execute_batch(cursor, depth_sql, depth_rows)

            conn.commit()
            return tick_id
    except Exception as e:
        logging.error(f"Error in insert_options_tick_data: {e}")
        logging.error(f"instrument_token: {tick_data.get('instrument_token')}")
        logging.error(f"strike: {instrument_info.get('strike')}, option_type: {instrument_info.get('instrument_type')}")
        import traceback
        logging.error(traceback.format_exc())
        conn.rollback()
        raise


def fetch_and_save_options():
    """Fetch and save all NIFTY options data from Kite API"""
    try:
        # Ensure database connection is initialized
        global conn
        try:
            if 'conn' not in globals() or conn is None:
                conn = get_db_connection()
            else:
                # Test if connection is still alive
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                except:
                    # Connection is dead, recreate it
                    try:
                        conn.close()
                    except:
                        pass
                    conn = get_db_connection()
        except Exception as db_err:
            logging.error(f"Database connection error: {db_err}")
            return {"success": False, "error": f"Database connection failed: {str(db_err)}", "fetched_count": 0, "total_symbols": 0}
        
        # Get all NFO instruments
        logging.info("Fetching NFO instruments from Kite API...")
        try:
            all_instruments = kite.instruments('NFO')
        except Exception as e:
            logging.error(f"Error fetching instruments from Kite: {e}")
            return {"success": False, "error": f"Kite API error: {str(e)}", "fetched_count": 0, "total_symbols": 0}
        
        # Filter for NIFTY options (CE and PE)
        nifty_options = []
        for inst in all_instruments:
            tradingsymbol = inst.get('tradingsymbol', '')
            instrument_type = inst.get('instrument_type')
            expiry = inst.get('expiry')
            
            if (tradingsymbol.startswith('NIFTY') 
                and instrument_type in ['CE', 'PE'] 
                and expiry is not None):
                # Convert expiry to date if it's a datetime object
                if isinstance(expiry, datetime):
                    expiry = expiry.date()
                elif isinstance(expiry, str):
                    try:
                        expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
                    except:
                        try:
                            expiry = datetime.strptime(expiry, '%Y-%m-%d %H:%M:%S').date()
                        except:
                            logging.warning(f"Could not parse expiry date: {expiry} for {tradingsymbol}")
                            continue
                
                # Store the converted expiry back in the instrument dict
                inst['expiry'] = expiry
                nifty_options.append(inst)
        
        logging.info(f"Found {len(nifty_options)} NIFTY options instruments")
        
        if not nifty_options:
            logging.warning("No NIFTY options found")
            return {"success": False, "error": "No NIFTY options found", "fetched_count": 0, "total_symbols": 0}
        
        # Create a mapping of instrument_token to instrument info for quick lookup
        instrument_map = {inst['instrument_token']: inst for inst in nifty_options}
        
        # Prepare symbols for batch quote fetching (Kite allows up to 2000 symbols per request)
        batch_size = 500  # Conservative batch size to avoid API limits
        total_symbols = len(nifty_options)
        fetched_count = 0
        failed_count = 0
        
        # Process in batches
        for i in range(0, total_symbols, batch_size):
            batch = nifty_options[i:i + batch_size]
            symbols = [f"NFO:{inst['tradingsymbol']}" for inst in batch]
            
            try:
                logging.info(f"Fetching quotes for batch {i//batch_size + 1} ({len(symbols)} symbols)...")
                
                # Fetch quotes for this batch
                quotes = kite.quote(symbols)
                
                # Process each quote
                for symbol in symbols:
                    tick_data = quotes.get(symbol, {})
                    
                    if not tick_data:
                        failed_count += 1
                        continue
                    
                    # Get instrument info
                    instrument_token = tick_data.get('instrument_token')
                    if not instrument_token or instrument_token not in instrument_map:
                        failed_count += 1
                        continue
                    
                    instrument_info = instrument_map[instrument_token]
                    
                    try:
                        # Add timestamp if not present
                        if 'timestamp' not in tick_data:
                            tick_data['timestamp'] = datetime.now()
                        
                        # Validate required fields from instrument_info
                        if not instrument_info.get('strike') or not instrument_info.get('instrument_type'):
                            logging.warning(f"Missing strike or instrument_type for {symbol}, skipping...")
                            failed_count += 1
                            continue
                        
                        # Insert into database
                        insert_options_tick_data(conn, tick_data, instrument_info)
                        fetched_count += 1
                        
                        # Log progress periodically
                        if fetched_count % 100 == 0:
                            logging.info(f"Processed {fetched_count} options so far...")
                            
                    except Exception as e:
                        logging.error(f"Error inserting tick data for {symbol}: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
                        failed_count += 1
                        continue
                
                # Small delay between batches to avoid rate limiting
                if i + batch_size < total_symbols:
                    time.sleep(0.5)
                    
            except Exception as e:
                logging.error(f"Error fetching batch {i//batch_size + 1}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                failed_count += len(batch)
                continue
        
        logging.info(f"Successfully fetched options data: {fetched_count} successful, {failed_count} failed out of {total_symbols} total")
        
        return {
            "success": True,
            "fetched_count": fetched_count,
            "failed_count": failed_count,
            "total_symbols": total_symbols
        }
        
    except Exception as e:
        logging.error(f"Error in fetch_and_save_options: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e), "fetched_count": 0, "total_symbols": 0}
    finally:
        # Close database connection
        if 'conn' in globals() and conn:
            try:
                conn.close()
            except:
                pass


# Allow running as script
if __name__ == "__main__":
    fetch_and_save_options()

