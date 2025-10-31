#!/usr/bin/env python3
"""
Standalone script to refresh holdings from Kite API
This script is called by cron every 5 minutes during market hours
"""

from Boilerplate import *
import logging
import sys
import traceback

def refresh_holdings():
    """Fetch and save current holdings from Kite API to database"""
    try:
        # Ensure database connection is initialized
        global conn, cursor
        try:
            if 'conn' not in globals() or conn is None:
                conn = get_db_connection()
                cursor = conn.cursor()
            else:
                # Try to check if connection is still alive
                try:
                    cursor.execute("SELECT 1")
                except:
                    # Connection is dead, recreate it
                    try:
                        conn.close()
                    except:
                        pass
                    conn = get_db_connection()
                    cursor = conn.cursor()
        except Exception as db_err:
            logging.error(f"Database connection error: {db_err}")
            logging.error(traceback.format_exc())
            print(f"✗ Database connection error: {db_err}")
            return
        
        # Ensure Kite connection is initialized
        try:
            from Boilerplate import get_access_token
            access_token = get_access_token()
            if not access_token:
                logging.error("No valid Kite access token available")
                print("✗ No valid Kite access token available")
                return
        except Exception as kite_err:
            logging.error(f"Kite initialization error: {kite_err}")
            print(f"✗ Kite initialization error: {kite_err}")
            return
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Fetching holdings from Kite...')
        holdings = kite.holdings()
        logging.info(f"User holdings: {holdings}")

        if not holdings:
            logging.warning("No holdings found")
            print("No holdings found")
            return

        valid_holdings = []
        for h in holdings:
            # Validate critical fields
            if not h.get('instrument_token'):
                logging.warning(f"Skipping holding with missing instrument_token: {h}")
                continue
            
            holding = {
                'tradingsymbol': h.get('tradingsymbol', ''),
                'instrument_token': h.get('instrument_token', 0),
                'isin': h.get('isin', ''),
                'quantity': h.get('quantity', 0),
                't1_quantity': h.get('t1_quantity', 0),
                'authorised_quantity': h.get('authorised_quantity', 0),
                'average_price': h.get('average_price', 0.0),
                'close_price': h.get('close_price', None),
                'last_price': h.get('last_price', None),
                'pnl': h.get('pnl', None),
                'collateral_quantity': h.get('collateral_quantity', 0),
                'collateral_type': h.get('collateral_type', None)
            }
            valid_holdings.append(holding)
            logging.debug(f"Processed holding: {holding}")
        
        if valid_holdings:
            cursor.executemany("""
                INSERT INTO my_schema.holdings (
                    trading_symbol, instrument_token,
                    isin, quantity, t1_quantity, authorised_quantity, average_price,
                    close_price, last_price, pnl, collateral_quantity, collateral_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (instrument_token, run_date) DO UPDATE
                SET trading_symbol = EXCLUDED.trading_symbol,
                    isin = EXCLUDED.isin,
                    quantity = EXCLUDED.quantity,
                    t1_quantity = EXCLUDED.t1_quantity,
                    authorised_quantity = EXCLUDED.authorised_quantity,
                    average_price = EXCLUDED.average_price,
                    close_price = EXCLUDED.close_price,
                    last_price = EXCLUDED.last_price,
                    pnl = EXCLUDED.pnl,
                    collateral_quantity = EXCLUDED.collateral_quantity,
                    collateral_type = EXCLUDED.collateral_type
            """, [
                (
                    h['tradingsymbol'], h['instrument_token'],
                    h['isin'], h['quantity'],
                    h['t1_quantity'], h['authorised_quantity'],
                    h['average_price'], h['close_price'], h['last_price'],
                    h['pnl'], h['collateral_quantity'], h['collateral_type']
                ) for h in valid_holdings
            ])
            conn.commit()
            logging.info(f"Stored {len(valid_holdings)} holdings")
            print(f"✓ Successfully updated {len(valid_holdings)} holdings")
        else:
            logging.warning("No valid holdings to save")
            print("No valid holdings to save")
            
    except Exception as e:
        if 'conn' in locals() and conn:
            try:
                conn.rollback()
            except:
                pass
        logging.error(f"Error refreshing holdings: {e}")
        logging.error(traceback.format_exc())
        print(f"✗ Error refreshing holdings: {e}")
        sys.exit(1)
    finally:
        # Close database connection
        if 'conn' in globals() and conn:
            try:
                cursor.close()
                conn.close()
            except:
                pass

if __name__ == "__main__":
    refresh_holdings()
    sys.exit(0)

