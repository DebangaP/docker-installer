import sys
import os
# Add parent directory to path to find common module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.Boilerplate import *

def insert_tick_data(conn, tick_data):
    with conn.cursor() as cursor:
        # Insert core tick data
        tick_sql = """
            INSERT INTO my_schema.futures_ticks 
            (instrument_token, timestamp, last_trade_time, last_price, last_quantity, buy_quantity, sell_quantity, volume,
             average_price, oi, oi_day_high, oi_day_low, net_change, lower_circuit_limit, upper_circuit_limit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """

        tick_values = (
            tick_data['instrument_token'],
            tick_data['timestamp'],
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
        )
        cursor.execute(tick_sql, tick_values)
        tick_id = cursor.fetchone()[0]

        # Insert OHLC data
        ohlc = tick_data.get('ohlc')
        if ohlc:
            ohlc_sql = """
                INSERT INTO my_schema.futures_tick_ohlc (tick_id, open, high, low, close)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(ohlc_sql, (tick_id, ohlc.get('open'), ohlc.get('high'), ohlc.get('low'), ohlc.get('close')))

        # Insert Depth data
        depth = tick_data.get('depth')
        if depth:
            depth_sql = """
                INSERT INTO my_schema.futures_tick_depth (tick_id, side, price, quantity, orders)
                VALUES (%s, %s, %s, %s, %s)
            """
            depth_rows = []
            for side in ['buy', 'sell']:
                for order in depth.get(side, []):
                    depth_rows.append((tick_id, side, order.get('price'), order.get('quantity'), order.get('orders')))
            if depth_rows:
                execute_batch(cursor, depth_sql, depth_rows)

        conn.commit()


def fetch_and_save_futures():
    """Fetch and save futures data for configured symbols"""
    try:
        symbols = ["NFO:NIFTY25OCTFUT"
                   ,"NFO:NIFTY25NOVFUT"
                   ,"NFO:NIFTY25DECFUT"]
        
        fetched_count = 0
        for i in range(len(symbols)):
            symbol = symbols[i]
            try:
                quote = kite.quote(symbol)
                tick_data = quote.get(symbol, {})
                
                if not tick_data:
                    logging.warning(f"No data returned for symbol {symbol}")
                    continue
                
                # Validate that we have required fields
                if 'instrument_token' not in tick_data:
                    logging.warning(f"No instrument_token for symbol {symbol}")
                    continue
                
                logging.info(f"Fetching futures data for {symbol}")
                logging.info(f"Last Price: {tick_data.get('last_price')}")
                logging.info(f"Volume: {tick_data.get('volume')}")
                logging.info(f"Open Interest (OI): {tick_data.get('oi')}")
                
                # Add timestamp to tick_data if not present
                if 'timestamp' not in tick_data:
                    from datetime import datetime
                    tick_data['timestamp'] = datetime.now()
                
                insert_tick_data(conn, tick_data)
                fetched_count += 1
            except Exception as e:
                logging.error(f"Error fetching futures data for {symbol}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        logging.info(f"Successfully fetched futures data for {fetched_count} symbols")
        return {"success": True, "fetched_count": fetched_count, "total_symbols": len(symbols)}
    except Exception as e:
        logging.error(f"Error in fetch_and_save_futures: {e}")
        return {"success": False, "error": str(e)}


# Allow running as script for backwards compatibility
if __name__ == "__main__":
    fetch_and_save_futures()
