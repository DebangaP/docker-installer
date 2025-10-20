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
                _insert_RT_OHLC = "INSERT INTO my_schema.RT_INTRADAY_PRICE (scrip_id, price_close, price_high, price_low, price_open, price_date, country, volume) VALUES( '" + _scrip_id + "', " + str(dailyrow.values[0])  + ", " + str(dailyrow.values[1])  + ", " + str(dailyrow.values[2])  + ", " + str(dailyrow.values[3]) + ",'" + date.strftime('%Y-%m-%d') + "', "+ _scrip_country + ", " + str(dailyrow.values[4]) + ")"

                try:
                    _DBconnection.execute(_insert_RT_OHLC)
                except errors.lookup(UNIQUE_VIOLATION) as uniqueKeyViolation:
                    print('Added Duplicate exists error in log table')
                    continue
                except Exception as e:
                    print('Exception -> ' + _scrip_id + ' ->' + str(e))
                    continue
                
    except Exception as exception:
        print('\n...Unable to fetch: '+ str(exception)) 
        
    session.close()
