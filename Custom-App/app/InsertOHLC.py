import pandas as pd
import yfinance as yf
#import psycopg2
from psycopg2 import errors
from psycopg2.errorcodes import UNIQUE_VIOLATION
import datetime
from datetime import timedelta
from time import sleep
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging

engine = create_engine('postgresql://postgres:postgres@postgres:5432/mydb')

_DBconnection = engine.connect()
Session = sessionmaker(bind=engine)
session = Session()

#SQLALCHEMY_SILENCE_UBER_WARNING=1

_fetch_last_update_dt = "SELECT coalesce(date(MAX(PRICE_DATE)) + 1, '2025-06-01') as \"last_upd_dt\" FROM my_schema.RT_INTRADAY_PRICE"

_df0 = pd.DataFrame(_DBconnection.execute(_fetch_last_update_dt), columns=['last_update_dt'])
for index, row in _df0.iterrows():
    _last_update_dt = str(row['last_update_dt'])

_current_dt = datetime.date.today() + timedelta(days = 1)   

#_last_update_dt = '2025-06-01'
#_current_dt = '2025-10-04'

_scrip_country = "'IN'"

logging.info('Start date, ' + str(_last_update_dt))
logging.info('Current date, ' + str(_current_dt))

if str(_last_update_dt) == str(_current_dt):
    logging.info("Exiting since latest data is available")
else:
    _fetchMissingPriceData = "SELECT SCRIP_ID, coalesce(YAHOO_CODE, scrip_id || '.NS') FROM my_schema.MASTER_SCRIPS"

    _scrip_id = ''
        
    try:
        print('Fetching data from DB')
        print(_fetchMissingPriceData)
        _df = pd.DataFrame(_DBconnection.execute(_fetchMissingPriceData), columns=['scrip_id', 'yahoo_code'])

        print('1')
        for index, row in _df.iterrows():
            _scrip_id = str(row['scrip_id'])
            _yahoo_code = str(row['yahoo_code'])
            print(_scrip_id)

            quote = yf.download(_yahoo_code, start=_last_update_dt, end=_current_dt)
            print(quote)
            
            for date, dailyrow in quote.iterrows():
                # Use ON CONFLICT to update existing records instead of failing
                _insert_RT_OHLC = """
                    INSERT INTO my_schema.RT_INTRADAY_PRICE 
                    (scrip_id, price_close, price_high, price_low, price_open, price_date, country, volume) 
                    VALUES (:param1, :param2, :param3, :param4, :param5, :param6, :param7, :param8)
                    ON CONFLICT (scrip_id, price_date) 
                    DO UPDATE SET 
                        price_close = EXCLUDED.price_close,
                        price_high = EXCLUDED.price_high,
                        price_low = EXCLUDED.price_low,
                        price_open = EXCLUDED.price_open,
                        country = EXCLUDED.country,
                        volume = EXCLUDED.volume
                """

                try:
                    from sqlalchemy import text
                    # Convert numpy types to Python native types
                    _DBconnection.execute(
                        text(_insert_RT_OHLC), 
                        {
                            'param1': str(_scrip_id),
                            'param2': float(dailyrow.values[0]),
                            'param3': float(dailyrow.values[1]),
                            'param4': float(dailyrow.values[2]),
                            'param5': float(dailyrow.values[3]),
                            'param6': date.strftime('%Y-%m-%d'),
                            'param7': 'IN',
                            'param8': int(dailyrow.values[4]) if not pd.isna(dailyrow.values[4]) else 0
                        }
                    )
                    print(f'Successfully inserted/updated data for {_scrip_id} on {date.strftime("%Y-%m-%d")}')
                except Exception as e:
                    print(f'Exception -> {_scrip_id} -> {str(e)}')
                    # SQLAlchemy Connection doesn't have rollback method on connection object
                    continue
                
    except Exception as exception:
        print('\n...Unable to fetch: '+ str(exception)) 
        
    session.close()
