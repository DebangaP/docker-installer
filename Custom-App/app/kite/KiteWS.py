from common.Boilerplate import *

kws = KiteTicker(API_KEY, access_token)

# Save tick data to PostgreSQL
def save_tick_to_db(tick):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        #logging.info(f"User holdings: {tick}")
        # Insert core tick data
        tick_sql = """
            INSERT INTO my_schema.ticks (instrument_token, timestamp, last_price, volume, oi, open, high, low, close)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        tick_data = (
            tick['instrument_token'],
            tick.get('timestamp') or datetime.now(),
            tick.get('last_price'),
            tick.get('volume'),
            tick.get('oi'),
            tick.get('ohlc', {}).get('open'),
            tick.get('ohlc', {}).get('high'),
            tick.get('ohlc', {}).get('low'),
            tick.get('ohlc', {}).get('close')
        )
        cursor.execute(tick_sql, tick_data)
        tick_id = cursor.fetchone()[0]

        # Insert depth data for MODE_FULL ticks
        if 'depth' in tick:
            depth_sql = """
                INSERT INTO my_schema.market_depth (tick_id, instrument_token, side, price, quantity, orders)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            depth_data = []
            for side in ['buy', 'sell']:
                for order in tick['depth'].get(side, [])[:5]:  # Up to 5 orders per side
                    depth_data.append((
                        tick_id,
                        tick['instrument_token'],
                        side,
                        order.get('price'),
                        order.get('quantity'),
                        order.get('orders')
                    ))
            if depth_data:
                execute_batch(cursor, depth_sql, depth_data)

        conn.commit()
        logging.debug(f"Saved tick for instrument {tick['instrument_token']}, tick_id: {tick_id}")
    except Exception as e:
        logging.error(f"Error saving tick to database: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

NIFTY25OCTFUT = 52168
NIFTY25NOVFUT = 37054
NIFTY_SPOT_TOKEN = 256265

def on_ticks(ws, ticks):
    # Callback to receive ticks.
    logging.debug("Ticks: {}".format(ticks))
    for tick in ticks:
        save_tick_to_db(tick)

def on_connect(ws, response):
    # Callback on successful connect.
    # Subscribe to a list of instrument_tokens.
    logging.info("Connected: {}".format(response))
    ws.subscribe([NIFTY_SPOT_TOKEN, NIFTY25OCTFUT, NIFTY25NOVFUT])
    ws.set_mode(ws.MODE_FULL, [NIFTY_SPOT_TOKEN, NIFTY25OCTFUT, NIFTY25NOVFUT])
    logging.info("Subscribed to tokens")

def on_close(ws, code, reason):
    # On connection close stop the main loop
    # Reconnection will not happen after executing `ws.stop()`
    ws.stop()

def on_error(ws, code, reason):
    # Handle errors
    logging.error(f"Error: {code}, {reason}")

# Assign the callbacks.
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close
kws.on_error = on_error

# Infinite loop on the main thread. Nothing after this will run.
# You have to use the pre-defined callbacks to manage subscriptions.
kws.connect()