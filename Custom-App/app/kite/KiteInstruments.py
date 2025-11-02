from common.Boilerplate import *

# Fetch all instruments
instruments = kite.instruments()
print(type(instruments[0]))

holdings = kite.holdings()
logging.info(holdings)

instrument_data = [
    tuple(instrument.get(k) for k in [
        'instrument_token','exchange_token','tradingsymbol','name',
        'last_price','expiry','strike','tick_size','lot_size',
        'instrument_type','segment','exchange'
    ])
    for instrument in instruments
    if (isinstance(instrument, dict) and instrument.get('expiry') is not None and str(instrument.get('expiry')).strip() != '')
]

#save to db
try:
    cursor.executemany("""
                INSERT INTO my_schema.instruments (
                   instrument_token, exchange_token, tradingsymbol, "name", 
                   last_price, expiry, strike, tick_size, lot_size, instrument_type, 
                   segment, exchange 
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, instrument_data)
    
    conn.commit()
except Exception as e:
    conn.rollback()
    logging.error(f"Error saving data: {e}")

