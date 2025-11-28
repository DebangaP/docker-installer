"""
Stocks Router
API endpoints for stock management (add, refresh data)
"""

from fastapi import APIRouter, Request
import os
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

router = APIRouter(prefix="/api", tags=["stocks"])
logger = logging.getLogger(__name__)


@router.post("/add_new_stock")
async def api_add_new_stock(request: Request):
    """API endpoint to add a new stock to master_scrips and fetch 6 months of historical data"""
    try:
        # Parse request body
        body = await request.json()
        symbol = (body.get('symbol') or '').strip().upper()
        yahoo_code = (body.get('yahoo_code') or '').strip()
        country = (body.get('country') or 'IN').strip() or 'IN'
        sector_code = (body.get('sector_code') or '').strip() or None
        
        # Validation
        if not symbol:
            return {"success": False, "error": "Stock symbol is required"}
        
        if not yahoo_code:
            return {"success": False, "error": "Yahoo Finance code is required"}
        
        # Get database config
        db_config = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'database': os.getenv('DB_NAME', 'mydb'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(connection_string)
        session = sessionmaker(bind=engine)()
        
        result = {
            'success': False,
            'message': '',
            'records_inserted': 0,
            'date_range': None,
            'error': None
        }
        
        try:
            # Check if stock already exists
            check_query = text("""
                SELECT scrip_id, yahoo_code FROM my_schema.master_scrips 
                WHERE scrip_id = :symbol AND scrip_country = :country
            """)
            existing = session.execute(check_query, {'symbol': symbol, 'country': country}).fetchone()
            
            stock_exists = existing is not None
            needs_data_fetch = False
            
            if stock_exists:
                # Check if there's any data in rt_intraday_price
                check_data_query = text("""
                    SELECT COUNT(*) as count FROM my_schema.rt_intraday_price 
                    WHERE scrip_id = :symbol AND country = :country
                """)
                data_count = session.execute(check_data_query, {'symbol': symbol, 'country': country}).fetchone()
                
                if data_count and data_count[0] > 0:
                    # Stock exists and has data - return error
                    session.close()
                    engine.dispose()
                    return {"success": False, "error": f"Stock {symbol} already exists in the database with historical data"}
                
                # Stock exists but no data - we'll update yahoo_code and fetch data
                needs_data_fetch = True
                existing_yahoo_code = existing[1] if existing else None
                
                # Update yahoo_code if it's different
                if existing_yahoo_code != yahoo_code:
                    update_query = text("""
                        UPDATE my_schema.master_scrips 
                        SET yahoo_code = :yahoo_code, updated_at = NOW()
                        WHERE scrip_id = :symbol AND scrip_country = :country
                    """)
                    session.execute(update_query, {'yahoo_code': yahoo_code, 'symbol': symbol, 'country': country})
                    session.commit()
                    logger.info(f"Updated yahoo_code for {symbol} from {existing_yahoo_code} to {yahoo_code}")
                else:
                    # Use existing yahoo_code
                    yahoo_code = existing_yahoo_code or yahoo_code
                    logger.info(f"Stock {symbol} exists with same yahoo_code {yahoo_code}, fetching historical data")
            else:
                # Add new stock to master_scrips
                insert_query = text("""
                    INSERT INTO my_schema.master_scrips 
                    (scrip_id, yahoo_code, scrip_country, sector_code, created_at, updated_at)
                    VALUES (:symbol, :yahoo_code, :country, :sector_code, NOW(), NOW())
                """)
                
                params = {
                    'symbol': symbol,
                    'yahoo_code': yahoo_code,
                    'country': country,
                    'sector_code': sector_code
                }
                
                session.execute(insert_query, params)
                session.commit()
                logger.info(f"Successfully added {symbol} to master_scrips")
                needs_data_fetch = True
            
            # Only fetch data if needed (new stock or existing stock without data)
            if not needs_data_fetch:
                session.close()
                engine.dispose()
                return {
                    "success": False,
                    "error": f"Stock {symbol} already exists in the database with historical data"
                }
            
            # Calculate date range for 6 months
            end_date = datetime.now().date() + timedelta(days=1)  # Today + 1 day (exclusive end)
            start_date = end_date - timedelta(days=180)  # Approximately 6 months (180 days)
            
            result['date_range'] = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
            
            # Fetch historical data from Yahoo Finance
            logger.info(f"Fetching 6 months of historical data for {symbol} ({yahoo_code}) from {start_date} to {end_date}")
            quote = yf.download(yahoo_code, start=start_date, end=end_date, progress=False)
            
            if quote.empty:
                session.close()
                engine.dispose()
                return {
                    "success": False,
                    "error": f"No historical data found for {yahoo_code}. Please verify the Yahoo Finance code is correct."
                }
            
            records_inserted = 0
            
            # Insert historical data into rt_intraday_price
            for date, dailyrow in quote.iterrows():
                insert_price_query = text("""
                    INSERT INTO my_schema.rt_intraday_price 
                    (scrip_id, price_close, price_high, price_low, price_open, price_date, country, volume) 
                    VALUES (:scrip_id, :close, :high, :low, :open, :date, :country, :volume)
                    ON CONFLICT (scrip_id, price_date) 
                    DO UPDATE SET 
                        price_close = EXCLUDED.price_close,
                        price_high = EXCLUDED.price_high,
                        price_low = EXCLUDED.price_low,
                        price_open = EXCLUDED.price_open,
                        country = EXCLUDED.country,
                        volume = EXCLUDED.volume,
                        created_at = CURRENT_TIMESTAMP
                """)
                
                try:
                    # Extract values from yfinance data
                    # yfinance returns DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
                    # Helper function to safely extract scalar value from Series or scalar
                    def safe_get_value(val):
                        """Safely extract scalar value, handling both Series and scalar inputs"""
                        if val is None:
                            return None
                        # If it's a Series, get the first value
                        if isinstance(val, pd.Series):
                            val = val.iloc[0] if len(val) > 0 else None
                        # Check if it's NaN
                        if val is None or (isinstance(val, (int, float)) and pd.isna(val)):
                            return None
                        return val
                    
                    try:
                        # Try to get values by column name
                        open_val = safe_get_value(dailyrow.get('Open'))
                        high_val = safe_get_value(dailyrow.get('High'))
                        low_val = safe_get_value(dailyrow.get('Low'))
                        close_val = safe_get_value(dailyrow.get('Close'))
                        volume_val = safe_get_value(dailyrow.get('Volume'))
                        
                        # Round prices to integers for consistent storage
                        open_price = round(float(open_val)) if open_val is not None else None
                        high_price = round(float(high_val)) if high_val is not None else None
                        low_price = round(float(low_val)) if low_val is not None else None
                        close_price = round(float(close_val)) if close_val is not None else None
                        volume_value = int(volume_val) if volume_val is not None else 0
                    except (KeyError, IndexError, AttributeError):
                        # Fallback to positional access if column names not available
                        # yfinance returns: Open, High, Low, Close, Adj Close, Volume
                        # So: values[0]=Open, values[1]=High, values[2]=Low, values[3]=Close, values[5]=Volume
                        try:
                            vals = dailyrow.values
                            open_val = safe_get_value(vals[0] if len(vals) > 0 else None)
                            high_val = safe_get_value(vals[1] if len(vals) > 1 else None)
                            low_val = safe_get_value(vals[2] if len(vals) > 2 else None)
                            close_val = safe_get_value(vals[3] if len(vals) > 3 else None)
                            volume_val = safe_get_value(vals[5] if len(vals) > 5 else None)
                            
                            open_price = round(float(open_val)) if open_val is not None else None
                            high_price = round(float(high_val)) if high_val is not None else None
                            low_price = round(float(low_val)) if low_val is not None else None
                            close_price = round(float(close_val)) if close_val is not None else None
                            volume_value = int(volume_val) if volume_val is not None else 0
                        except (IndexError, ValueError, TypeError):
                            # If all else fails, skip this row
                            logger.warning(f"Could not extract price data for {symbol} on {date}")
                            continue
                    
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
                    
                    session.execute(
                        insert_price_query,
                        {
                            'scrip_id': symbol,
                            'close': close_price,
                            'high': high_price,
                            'low': low_price,
                            'open': open_price,
                            'date': date_str,
                            'country': country,
                            'volume': volume_value
                        }
                    )
                    records_inserted += 1 #no of records inserted
                    
                except Exception as e:
                    logger.warning(f"Error inserting price data for {symbol} on {date}: {str(e)}")
                    continue
            
            session.commit()
            
            result['success'] = True
            # Determine if this was a new stock or existing stock update
            if stock_exists:
                result['message'] = f"Stock {symbol} already existed in database. Updated yahoo_code and fetched {records_inserted} historical records"
            else:
                result['message'] = f"Successfully added {symbol} to database and fetched {records_inserted} historical records"
            result['records_inserted'] = records_inserted
            
            logger.info(f"Successfully {'updated' if stock_exists else 'added'} {symbol} with {records_inserted} historical records")
            
        except Exception as e:
            error_msg = f"Error adding stock {symbol}: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            result['error'] = error_msg
            result['success'] = False
            result['message'] = error_msg
            
        finally:
            session.close()
            engine.dispose()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in add_new_stock API: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


@router.post("/refresh_stock_data")
async def api_refresh_stock_data(request: Request):
    """API endpoint to delete and re-fetch stock price data (useful after stock splits/bonus issues)"""
    try:
        # Parse request body
        body = await request.json()
        symbol = (body.get('symbol') or '').strip().upper()
        country = (body.get('country') or 'IN').strip() or 'IN'
        
        # Validation
        if not symbol:
            return {"success": False, "error": "Stock symbol is required"}
        
        # Get database config
        db_config = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'database': os.getenv('DB_NAME', 'mydb'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(connection_string)
        session = sessionmaker(bind=engine)()
        
        result = {
            'success': False,
            'message': '',
            'records_deleted': 0,
            'records_inserted': 0,
            'date_range': None,
            'error': None
        }
        
        try:
            # Check if stock exists in master_scrips
            check_query = text("""
                SELECT scrip_id, yahoo_code FROM my_schema.master_scrips 
                WHERE scrip_id = :symbol AND scrip_country = :country
            """)
            existing = session.execute(check_query, {'symbol': symbol, 'country': country}).fetchone()
            
            if not existing:
                session.close()
                engine.dispose()
                return {
                    "success": False,
                    "error": f"Stock {symbol} not found in master_scrips. Please add it first using 'Add New Stock'."
                }
            
            yahoo_code = existing[1]
            if not yahoo_code:
                session.close()
                engine.dispose()
                return {
                    "success": False,
                    "error": f"Yahoo Finance code not found for {symbol}. Please update master_scrips with yahoo_code first."
                }
            
            # Delete all existing data from rt_intraday_price for this stock
            delete_query = text("""
                DELETE FROM my_schema.rt_intraday_price 
                WHERE scrip_id = :symbol AND country = :country
            """)
            delete_result = session.execute(delete_query, {'symbol': symbol, 'country': country})
            session.commit()
            records_deleted = delete_result.rowcount
            result['records_deleted'] = records_deleted
            
            logger.info(f"Deleted {records_deleted} records for {symbol} from rt_intraday_price")
            
            # Calculate date range for 6 months
            end_date = datetime.now().date() + timedelta(days=1)  # Today + 1 day (exclusive end)
            start_date = end_date - timedelta(days=180)  # Approximately 6 months (180 days)
            
            result['date_range'] = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
            
            # Fetch historical data from Yahoo Finance
            logger.info(f"Fetching 6 months of historical data for {symbol} ({yahoo_code}) from {start_date} to {end_date}")
            quote = yf.download(yahoo_code, start=start_date, end=end_date, progress=False)
            
            if quote.empty:
                session.close()
                engine.dispose()
                return {
                    "success": False,
                    "error": f"No historical data found for {yahoo_code}. Please verify the Yahoo Finance code is correct."
                }
            
            records_inserted = 0
            
            # Insert historical data into rt_intraday_price
            for date, dailyrow in quote.iterrows():
                insert_price_query = text("""
                    INSERT INTO my_schema.rt_intraday_price 
                    (scrip_id, price_close, price_high, price_low, price_open, price_date, country, volume) 
                    VALUES (:scrip_id, :close, :high, :low, :open, :date, :country, :volume)
                    ON CONFLICT (scrip_id, price_date) 
                    DO UPDATE SET 
                        price_close = EXCLUDED.price_close,
                        price_high = EXCLUDED.price_high,
                        price_low = EXCLUDED.price_low,
                        price_open = EXCLUDED.price_open,
                        country = EXCLUDED.country,
                        volume = EXCLUDED.volume,
                        created_at = CURRENT_TIMESTAMP
                """)
                
                try:
                    # Extract values from yfinance data
                    # yfinance returns DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
                    # Helper function to safely extract scalar value from Series or scalar
                    def safe_get_value(val):
                        """Safely extract scalar value, handling both Series and scalar inputs"""
                        if val is None:
                            return None
                        # If it's a Series, get the first value
                        if isinstance(val, pd.Series):
                            val = val.iloc[0] if len(val) > 0 else None
                        # Check if it's NaN
                        if val is None or (isinstance(val, (int, float)) and pd.isna(val)):
                            return None
                        return val
                    
                    try:
                        # Try to get values by column name
                        open_val = safe_get_value(dailyrow.get('Open'))
                        high_val = safe_get_value(dailyrow.get('High'))
                        low_val = safe_get_value(dailyrow.get('Low'))
                        close_val = safe_get_value(dailyrow.get('Close'))
                        volume_val = safe_get_value(dailyrow.get('Volume'))
                        
                        # Round prices to integers for consistent storage
                        open_price = round(float(open_val)) if open_val is not None else None
                        high_price = round(float(high_val)) if high_val is not None else None
                        low_price = round(float(low_val)) if low_val is not None else None
                        close_price = round(float(close_val)) if close_val is not None else None
                        volume_value = int(volume_val) if volume_val is not None else 0
                    except (KeyError, IndexError, AttributeError):
                        # Fallback to positional access if column names not available
                        # yfinance returns: Open, High, Low, Close, Adj Close, Volume
                        # So: values[0]=Open, values[1]=High, values[2]=Low, values[3]=Close, values[5]=Volume
                        try:
                            vals = dailyrow.values
                            open_val = safe_get_value(vals[0] if len(vals) > 0 else None)
                            high_val = safe_get_value(vals[1] if len(vals) > 1 else None)
                            low_val = safe_get_value(vals[2] if len(vals) > 2 else None)
                            close_val = safe_get_value(vals[3] if len(vals) > 3 else None)
                            volume_val = safe_get_value(vals[5] if len(vals) > 5 else None)
                            
                            open_price = round(float(open_val)) if open_val is not None else None
                            high_price = round(float(high_val)) if high_val is not None else None
                            low_price = round(float(low_val)) if low_val is not None else None
                            close_price = round(float(close_val)) if close_val is not None else None
                            volume_value = int(volume_val) if volume_val is not None else 0
                        except (IndexError, ValueError, TypeError):
                            # If all else fails, skip this row
                            logger.warning(f"Could not extract price data for {symbol} on {date}")
                            continue
                    
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
                    
                    session.execute(
                        insert_price_query,
                        {
                            'scrip_id': symbol,
                            'close': close_price,
                            'high': high_price,
                            'low': low_price,
                            'open': open_price,
                            'date': date_str,
                            'country': country,
                            'volume': volume_value
                        }
                    )
                    records_inserted += 1
                    
                except Exception as e:
                    logger.warning(f"Error inserting price data for {symbol} on {date}: {str(e)}")
                    continue
            
            session.commit()
            
            result['success'] = True
            result['message'] = f"Successfully refreshed data for {symbol}. Deleted {records_deleted} old records and inserted {records_inserted} new records"
            result['records_inserted'] = records_inserted
            
            logger.info(f"Successfully refreshed {symbol}: deleted {records_deleted} records, inserted {records_inserted} records")
            
        except Exception as e:
            error_msg = f"Error refreshing stock data for {symbol}: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            result['error'] = error_msg
            result['success'] = False
            result['message'] = error_msg
            
        finally:
            session.close()
            engine.dispose()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in refresh_stock_data API: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


@router.post("/derivatives_suggestions/calculate_theoretical_pnl")
async def api_calculate_theoretical_pnl(request: Request):
    """API endpoint to calculate theoretical P&L for PENDING derivative suggestions"""
    try:
        from api.services.theoretical_pnl_calculator import TheoreticalPnlCalculator
        
        body = await request.json()
        start_date = body.get('start_date')
        end_date = body.get('end_date')
        strategy_type = body.get('strategy_type')
        source = body.get('source')
        
        calculator = TheoreticalPnlCalculator()
        result = calculator.calculate_theoretical_pnl_for_suggestions(
            start_date=start_date,
            end_date=end_date,
            strategy_type=strategy_type,
            source=source
        )
        
        return {
            "success": True,
            "updated_count": result['updated_count'],
            "suggestions_processed": result['suggestions_processed'],
            "message": f"Calculated theoretical P&L for {result['updated_count']} out of {result['suggestions_processed']} suggestions"
        }
        
    except Exception as e:
        logger.error(f"Error calculating theoretical P&L: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


@router.post("/derivatives_suggestions/monitor")
async def api_monitor_derivative_suggestions():
    """API endpoint to manually trigger monitoring of MOCKED derivative suggestions"""
    try:
        from api.services.derivative_suggestions_monitor import DerivativeSuggestionsMonitor
        
        monitor = DerivativeSuggestionsMonitor()
        result = monitor.monitor_and_exit_suggestions()
        
        if 'error' in result:
            return {
                "success": False,
                "error": result['error']
            }
        
        return {
            "success": True,
            "statistics": result,
            "message": f"Monitored {result.get('total_checked', 0)} suggestions. Exits: {result.get('target_exits', 0) + result.get('stop_loss_exits', 0) + result.get('conflicting_signal_exits', 0)}"
        }
        
    except Exception as e:
        logger.error(f"Error monitoring derivative suggestions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

