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


def refresh_stock_prices(db_config=None):
    """
    Refresh stock price data from Yahoo Finance and update database
    
    Args:
        db_config: Optional database configuration dict. If None, uses default config.
        
    Returns:
        dict: Result dictionary with success status, message, and statistics
    """
    if db_config is None:
        db_config = {
            'host': 'postgres',
            'database': 'mydb',
            'user': 'postgres',
            'password': 'postgres',
            'port': 5432
        }
    
    connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    engine = create_engine(connection_string)
    _DBconnection = engine.connect()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    result = {
        'success': False,
        'message': '',
        'stocks_processed': 0,
        'records_inserted': 0,
        'errors': []
    }
    
    try:
        from sqlalchemy import text
        
        # Get last update date
        _fetch_last_update_dt = "SELECT coalesce(date(MAX(PRICE_DATE)) + 1, '2025-06-01') as \"last_upd_dt\" FROM my_schema.RT_INTRADAY_PRICE"
        _df0 = pd.DataFrame(_DBconnection.execute(text(_fetch_last_update_dt)), columns=['last_update_dt'])
        for index, row in _df0.iterrows():
            _last_update_dt = str(row['last_update_dt'])
        
        _current_dt = datetime.date.today() + timedelta(days=1)
        
        logging.info('Start date, ' + str(_last_update_dt))
        logging.info('Current date, ' + str(_current_dt))
        
        if str(_last_update_dt) == str(_current_dt):
            result['success'] = True
            result['message'] = 'Latest data is already available. No update needed.'
            logging.info("Exiting since latest data is available")
            return result
        
        # Fetch stock data from master_scrips
        _fetchMissingPriceData = "SELECT SCRIP_ID, coalesce(YAHOO_CODE, scrip_id || '.NS') FROM my_schema.MASTER_SCRIPS"
        _df = pd.DataFrame(_DBconnection.execute(text(_fetchMissingPriceData)), columns=['scrip_id', 'yahoo_code'])
        
        stocks_processed = 0
        records_inserted = 0
        
        for index, row in _df.iterrows():
            _scrip_id = str(row['scrip_id'])
            _yahoo_code = str(row['yahoo_code'])
            
            try:
                quote = yf.download(_yahoo_code, start=_last_update_dt, end=_current_dt, progress=False)
                
                if quote.empty:
                    continue
                
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
                            volume = EXCLUDED.volume,
                            created_at = CURRENT_TIMESTAMP
                    """
                    
                    try:
                        # Convert numpy types to Python native types and round prices to integers
                        _DBconnection.execute(
                            text(_insert_RT_OHLC), 
                            {
                                'param1': str(_scrip_id),
                                'param2': round(float(dailyrow.values[0])) if not pd.isna(dailyrow.values[0]) else None,
                                'param3': round(float(dailyrow.values[1])) if not pd.isna(dailyrow.values[1]) else None,
                                'param4': round(float(dailyrow.values[2])) if not pd.isna(dailyrow.values[2]) else None,
                                'param5': round(float(dailyrow.values[3])) if not pd.isna(dailyrow.values[3]) else None,
                                'param6': date.strftime('%Y-%m-%d'),
                                'param7': 'IN',
                                'param8': int(dailyrow.values[4]) if len(dailyrow.values) > 4 and not pd.isna(dailyrow.values[4]) else 0
                            }
                        )
                        records_inserted += 1
                        logging.info(f'Successfully inserted/updated data for {_scrip_id} on {date.strftime("%Y-%m-%d")}')
                    except Exception as e:
                        error_msg = f'Exception -> {_scrip_id} -> {str(e)}'
                        result['errors'].append(error_msg)
                        logging.error(error_msg)
                        continue
                
                stocks_processed += 1
                
            except Exception as e:
                error_msg = f'Error fetching data for {_scrip_id}: {str(e)}'
                result['errors'].append(error_msg)
                logging.error(error_msg)
                continue
        
        result['success'] = True
        result['stocks_processed'] = stocks_processed
        result['records_inserted'] = records_inserted
        result['message'] = f'Successfully processed {stocks_processed} stocks and inserted/updated {records_inserted} records.'
        
    except Exception as exception:
        error_msg = f'Unable to refresh stock prices: {str(exception)}'
        result['message'] = error_msg
        result['errors'].append(error_msg)
        logging.error(error_msg)
        
    finally:
        session.close()
        _DBconnection.close()
        engine.dispose()
    
    return result


# Main execution when run as script (maintain backward compatibility)
if __name__ == '__main__':
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
                            volume = EXCLUDED.volume,
                            created_at = CURRENT_TIMESTAMP
                    """
    
                    try:
                        from sqlalchemy import text
                        # Convert numpy types to Python native types and round prices to integers
                        _DBconnection.execute(
                            text(_insert_RT_OHLC), 
                            {
                                'param1': str(_scrip_id),
                                'param2': round(float(dailyrow.values[0])),
                                'param3': round(float(dailyrow.values[1])),
                                'param4': round(float(dailyrow.values[2])),
                                'param5': round(float(dailyrow.values[3])),
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
