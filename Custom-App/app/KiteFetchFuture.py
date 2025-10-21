from Boilerplate import *

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


symbols = ["NFO:NIFTY25OCTFUT"
           ,"NFO:NIFTY25NOVFUT"
           ,"NFO:NIFTY25DECFUT"]

for i in range(len(symbols)):
    symbol = symbols[i]
    quote = kite.quote(symbol)
    tick_data = quote[symbol]
    logging.info(f"Last Price: {tick_data['last_price']}")
    logging.info(f"Volume: {tick_data['volume']}")
    logging.info(f"Open Interest (OI): {tick_data['oi']}")
    logging.info(tick_data)

    insert_tick_data(conn, tick_data)
