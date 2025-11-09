"""
Fundamentals Fetcher
Fetches fundamental data from Yahoo Finance and stores in database
"""

import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2 import errors as psycopg2_errors
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional
from common.Boilerplate import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YahooFundamentalsFetcher:
    """
    Fetches fundamental data (actions, income statements, insider transactions) 
    from Yahoo Finance and stores in database
    """
    
    def __init__(self):
        self._cancellation_flag_key = "yahoo_fundamentals_fetch_cancelled"
        # Initialize Redis client once for reuse
        self._redis_client = None
        self._init_redis_client()
        # Ensure database table exists
        self._ensure_system_flags_table()
        
        # Excluded stocks that cause errors
        self.excluded_stocks = [
            'RMCSWITCH', 'V-MARC', 'JKLAKSHMI', 'JYOTHYLAB', 'KARURVYSYA', 'INDUSTOWER', 
            'JSWENERGY', 'IOC', 'SHREYANIND', 'HINDCOPPER', 'MMTC', 'LT', 'LTI', 
            'MAHABANK', 'MARICO', 'MHRIL', 'MIDHANI', 'MRPL', 'NATIONALUM', 'NESTLEIND', 
            'NETWORK18', 'NOCIL', 'NTPC', 'M_M', 'OIL', 'LINDEINDIA', 'M&MFIN', 
            'MAZDOCK', 'OBEROIRLTY', 'MGL', 'PETRONET', 'KSB', 'PGHL', 'PNCINFRA', 
            'RACLGEAR', 'PERMAGN', 'KAYNES', 'POWERGRID', 'RAYMOND', 'RCF', 'RESPONIND', 
            'ROSSARI', 'ROUTE', 'RPGLIFE', 'SANOFI', 'SARDAEN', 'SBIN', 'SCHNEIDER', 
            'SCI', 'SHARDACROP', 'SONATSOFTW', 'GRSE', 'AKZOINDIA', 'AMARAJABAT', 
            'AMBUJACEM', 'BALAMINES', 'BHARATRAS', 'BALKRISIND', 'PFIZER', 'PRESTIGE', 
            'RAJESHEXPO', 'SKFINDIA', 'SOLARINDS', 'RHIM', 'PNB', 'RITES', 'GNFC', 
            'LAXMIMACH', 'APARINDS', 'BALMLAWRIE', 'FIVESTAR', 'FUSION', 'GAIL', 
            'DYCL', 'BSE', 'BEL', 'FRETAIL', 'UCOBANK', 'ABBOTINDIA', 'GICRE', 
            'WIPRO', 'TATACOFFEE', 'TATAINVEST', 'VARROC', 'WELSPUNIND', 'WHIRLPOOL', 
            'KOTHARIPET', 'CASTROLIND', 'INOXLEISUR', 'MINDTREE', 'ANGELBRKG', 
            'ADANITRANS', 'SHRIRAMCIT', 'MAHINDCIE', 'BANKINDIA', 'SUPREMEIND', 
            'RAIN', 'COALINDIA', 'VBL', 'ADANIGREEN', 'ASAHIINDIA', 'DCBBANK', 
            'JSLHISAR', 'PVR', 'MINDAIND', 'HDFC', 'DAAWAT', 'DATAPATTNS', 
            'CADILAHC', 'IIFLWAM', 'SPICEJET', 'SRTRANSFIN', 'ACC', 'SUNDRMFAST', 
            'INDHOTEL', 'MCX', 'KALPATPOWR', 'TASTYBITE', 'CDSL', 'EQUITAS', 
            'SUNCLAYLTD', 'CHALET', 'MOTILALOFS', 'NLCINDIA', 'NSLNISP', 
            'POWERINDIA', 'HAL', 'SHILCHAR', 'L_TFH', 'NETWEB', 'SUMICHEM', 
            'GLAXO', 'IDFCFIRSTB', 'NAVINFLUOR', 'SUNDARMFIN', 'FACT', 'NIACL', 
            'MSUMI', 'LODHA', 'SIS', 'SUZLON'
        ]
    
    def _init_redis_client(self):
        """Initialize Redis client for cancellation flag"""
        try:
            import redis
            self._redis_client = redis.Redis(host='redis', port=6379, decode_responses=True, socket_connect_timeout=2)
            # Test connection
            self._redis_client.ping()
            logger.debug("Redis client initialized successfully")
        except Exception as e:
            logger.debug(f"Redis not available, will use database: {e}")
            self._redis_client = None
    
    def _ensure_system_flags_table(self):
        """Ensure system_flags table exists in database"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS my_schema.system_flags (
                    flag_key VARCHAR(100) PRIMARY KEY,
                    value VARCHAR(10),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.warning(f"Error ensuring system_flags table exists: {e}")
    
    def check_cancellation(self) -> bool:
        """
        Check if batch processing has been cancelled by user
        
        Returns:
            True if cancelled, False otherwise
        """
        try:
            # Try Redis first (faster)
            if self._redis_client:
                try:
                    cancelled = self._redis_client.get(self._cancellation_flag_key)
                    if cancelled and (cancelled == '1' or cancelled == 'true'):
                        return True
                except Exception as e:
                    logger.debug(f"Redis check failed, trying database: {e}")
                    # Reinitialize Redis client
                    self._init_redis_client()
            
            # Fallback to database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM my_schema.system_flags 
                WHERE flag_key = %s
            """, (self._cancellation_flag_key,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            if result and result[0]:
                cancelled_value = result[0]
                if cancelled_value == '1' or cancelled_value == 'true':
                    return True
            return False
        except Exception as e:
            logger.debug(f"Error checking cancellation flag: {e}")
            return False
    
    def set_cancellation_flag(self, cancelled: bool = True):
        """
        Set cancellation flag for batch processing
        
        Args:
            cancelled: True to cancel, False to clear cancellation
        """
        try:
            # Try Redis first
            if self._redis_client:
                try:
                    if cancelled:
                        self._redis_client.set(self._cancellation_flag_key, '1', ex=3600)  # Expire after 1 hour
                        logger.info(f"Cancellation flag set in Redis with key: {self._cancellation_flag_key}")
                    else:
                        # When clearing, delete from Redis
                        deleted = self._redis_client.delete(self._cancellation_flag_key)
                        if deleted:
                            logger.info(f"Cancellation flag cleared from Redis (key: {self._cancellation_flag_key})")
                        else:
                            logger.debug(f"Cancellation flag not found in Redis (key: {self._cancellation_flag_key})")
                except Exception as e:
                    logger.warning(f"Redis operation failed, using database only: {e}")
                    self._init_redis_client()
            
            # Always set in database as well (for reliability)
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO my_schema.system_flags (flag_key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (flag_key) 
                DO UPDATE SET value = %s, updated_at = NOW()
            """, (self._cancellation_flag_key, '1' if cancelled else '0', '1' if cancelled else '0'))
            conn.commit()
            cursor.close()
            conn.close()
            if cancelled:
                logger.info(f"Cancellation flag set in database (key: {self._cancellation_flag_key})")
            else:
                logger.info(f"Cancellation flag cleared in database (key: {self._cancellation_flag_key})")
        except Exception as e:
            logger.error(f"Error setting cancellation flag: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def fetch_fundamentals_for_stock(self, scrip_id: str, yahoo_code: str) -> Dict:
        """
        Fetch fundamental data for a single stock
        
        Args:
            scrip_id: Stock symbol
            yahoo_code: Yahoo Finance code
            
        Returns:
            Dictionary with status and results
        """
        result = {
            'scrip_id': scrip_id,
            'status': 'success',
            'actions_count': 0,
            'income_count': 0,
            'insider_count': 0
        }
        
        try:
            # Fetch data from Yahoo Finance
            yf_ticker = yf.Ticker(yahoo_code)
            
            # 1. Fetch and store actions (dividends, splits)
            try:
                actions = yf_ticker.actions
                if actions is not None and not actions.empty:
                    actions.reset_index(inplace=True)
                    
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    
                    for _, row in actions.iterrows():
                        try:
                            insert_actions = """
                                INSERT INTO my_schema.rt_scrip_actions 
                                (scrip_id, dividend, split, trans_date) 
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT DO NOTHING
                            """
                            cursor.execute(insert_actions, (
                                scrip_id,
                                float(row.get('Dividends', 0)) if pd.notna(row.get('Dividends')) else 0,
                                float(row.get('Stock Splits', 0)) if pd.notna(row.get('Stock Splits')) else 0,
                                row['Date'].date() if hasattr(row['Date'], 'date') else row['Date']
                            ))
                            result['actions_count'] += 1
                        except psycopg2_errors.UniqueViolation:
                            # Data already exists, skip
                            continue
                        except Exception as e:
                            logger.debug(f"Error inserting action for {scrip_id}: {e}")
                            continue
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
            except Exception as e:
                logger.warning(f"Error fetching actions for {scrip_id}: {e}")
            
            # 2. Fetch and store quarterly income statements
            try:
                income_stmt = yf_ticker.quarterly_income_stmt
                if income_stmt is not None and not income_stmt.empty:
                    stmt_columns = income_stmt.columns
                    
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    
                    for col in stmt_columns[:5]:  # Process first 5 quarters
                        temp_series = income_stmt[col]
                        temp_df = temp_series.to_frame()
                        temp_df.reset_index(inplace=True)
                        
                        for _, row in temp_df.iterrows():
                            try:
                                insert_income = """
                                    INSERT INTO my_schema.rt_quarterly_income 
                                    (scrip_id, trans_date, trans_tag, trans_value) 
                                    VALUES (%s, %s, %s, %s)
                                    ON CONFLICT DO NOTHING
                                """
                                trans_date = col.date() if hasattr(col, 'date') else col
                                trans_tag = str(row.iloc[0]) if len(row) > 0 else ''
                                trans_value = str(row.iloc[1]) if len(row) > 1 else '0'
                                
                                cursor.execute(insert_income, (
                                    scrip_id,
                                    trans_date,
                                    trans_tag,
                                    trans_value
                                ))
                                result['income_count'] += 1
                            except psycopg2_errors.UniqueViolation:
                                continue
                            except Exception as e:
                                logger.debug(f"Error inserting income for {scrip_id}: {e}")
                                continue
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
            except Exception as e:
                logger.warning(f"Error fetching income statement for {scrip_id}: {e}")
            
            # 3. Fetch and store insider transactions
            try:
                insider_trans = yf_ticker.insider_transactions
                if insider_trans is not None and not insider_trans.empty:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    
                    for _, row in insider_trans.iterrows():
                        try:
                            insert_insider = """
                                INSERT INTO my_schema.rt_insider_trans 
                                (scrip_id, shares, value, text, insider, position, trans_date, ownership) 
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT DO NOTHING
                            """
                            
                            trans_date = row.get('TransactionDate', None)
                            if trans_date:
                                if hasattr(trans_date, 'date'):
                                    trans_date = trans_date.date()
                                elif isinstance(trans_date, str):
                                    trans_date = datetime.strptime(trans_date[:10], '%Y-%m-%d').date()
                            
                            cursor.execute(insert_insider, (
                                scrip_id,
                                str(row.get('Shares', '')) if pd.notna(row.get('Shares')) else '',
                                str(row.get('Value', '')) if pd.notna(row.get('Value')) else '',
                                str(row.get('Text', '')) if pd.notna(row.get('Text')) else '',
                                str(row.get('Insider', '')) if pd.notna(row.get('Insider')) else '',
                                str(row.get('Position', '')) if pd.notna(row.get('Position')) else '',
                                trans_date,
                                str(row.get('Ownership', '')) if pd.notna(row.get('Ownership')) else ''
                            ))
                            result['insider_count'] += 1
                        except psycopg2_errors.UniqueViolation:
                            continue
                        except Exception as e:
                            logger.debug(f"Error inserting insider transaction for {scrip_id}: {e}")
                            continue
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
            except Exception as e:
                logger.warning(f"Error fetching insider transactions for {scrip_id}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {scrip_id}: {e}")
            return {
                'scrip_id': scrip_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def fetch_fundamentals_for_stocks(self, stocks: List[tuple], batch_size: int = 5, delay_between_batches: float = 3.45) -> List[Dict]:
        """
        Fetch fundamental data for a list of stocks in batches
        
        Args:
            stocks: List of tuples (scrip_id, yahoo_code)
            batch_size: Number of stocks to process in each batch
            delay_between_batches: Delay in seconds between batches
            
        Returns:
            List of results with status for each stock
        """
        results = []
        success_count = 0
        error_count = 0
        
        # Clear cancellation flag at start
        self.set_cancellation_flag(False)
        
        # Process in batches
        cancelled = False
        for batch_idx in range(0, len(stocks), batch_size):
            # Check for cancellation before each batch
            if self.check_cancellation():
                logger.info("Yahoo fundamentals fetch cancelled by user")
                results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                cancelled = True
                break
            
            batch = stocks[batch_idx:batch_idx + batch_size]
            logger.info(f"Processing batch {batch_idx // batch_size + 1} ({len(batch)} stocks)...")
            
            for scrip_id, yahoo_code in batch:
                # Check for cancellation before each stock
                if self.check_cancellation():
                    logger.info("Yahoo fundamentals fetch cancelled by user")
                    results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                    cancelled = True
                    break
                
                try:
                    logger.info(f"Processing {scrip_id} ({yahoo_code})...")
                    
                    result = self.fetch_fundamentals_for_stock(scrip_id, yahoo_code)
                    results.append(result)
                    
                    if result.get('status') == 'success':
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {scrip_id}: {e}")
                    error_count += 1
                    results.append({
                        'scrip_id': scrip_id,
                        'status': 'failed',
                        'error': str(e)
                    })
                    continue
                
                # Check for cancellation after each stock
                if self.check_cancellation():
                    logger.info("Yahoo fundamentals fetch cancelled by user")
                    results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                    cancelled = True
                    break
                
                # Rate limiting - be respectful to Yahoo Finance
                # Check cancellation during delay
                if delay_between_batches > 0:
                    delay_elapsed = 0
                    while delay_elapsed < delay_between_batches:
                        if self.check_cancellation():
                            logger.info("Yahoo fundamentals fetch cancelled by user")
                            results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                            cancelled = True
                            break
                        time.sleep(0.1)
                        delay_elapsed += 0.1
                    if cancelled:
                        break
            
            # Check for cancellation after batch
            if cancelled:
                break
            
            # Delay between batches (if delay > 0 and not last batch)
            if not cancelled and batch_idx + batch_size < len(stocks) and delay_between_batches > 0:
                # Check cancellation during delay
                delay_elapsed = 0
                while delay_elapsed < delay_between_batches:
                    if self.check_cancellation():
                        logger.info("Yahoo fundamentals fetch cancelled by user")
                        results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                        cancelled = True
                        break
                    time.sleep(0.1)
                    delay_elapsed += 0.1
                if cancelled:
                    break
        
        logger.info(f"Yahoo fundamentals fetch completed: {success_count} succeeded, {error_count} failed")
        return results
    
    def fetch_fundamentals_for_holdings(self, batch_size: int = 5, delay_between_batches: float = 3.45) -> List[Dict]:
        """
        Fetch fundamental data for all holdings
        
        Args:
            batch_size: Number of stocks to process in each batch
            delay_between_batches: Delay in seconds between batches
            
        Returns:
            List of results with status for each stock
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all holdings with yahoo_code
            cursor.execute("""
                SELECT DISTINCT ms.scrip_id, COALESCE(ms.yahoo_code, ms.scrip_id || '.NS') as yahoo_code
                FROM my_schema.master_scrips ms
                INNER JOIN my_schema.holdings h ON h.trading_symbol = ms.scrip_id
                WHERE ms.scrip_country = 'IN'
                AND ms.scrip_id NOT IN (SELECT unnest(%s::text[]))
            """, (self.excluded_stocks,))
            
            stocks = cursor.fetchall()
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(stocks)} holdings to fetch Yahoo fundamentals for")
            
            if not stocks:
                return []
            
            return self.fetch_fundamentals_for_stocks(stocks, batch_size, delay_between_batches)
            
        except Exception as e:
            logger.error(f"Error getting holdings for Yahoo fundamentals fetch: {e}")
            return []
    
    def fetch_fundamentals_for_nifty50(self, batch_size: int = 5, delay_between_batches: float = 3.45) -> List[Dict]:
        """
        Fetch fundamental data for Nifty50 stocks
        
        Args:
            batch_size: Number of stocks to process in each batch
            delay_between_batches: Delay in seconds between batches
            
        Returns:
            List of results with status for each stock
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get Nifty50 stocks (exclude the index itself and excluded stocks)
            cursor.execute("""
                SELECT scrip_id, COALESCE(yahoo_code, scrip_id || '.NS') as yahoo_code
                FROM my_schema.master_scrips
                WHERE scrip_country = 'IN'
                AND (scrip_group LIKE '%NIFTY50%' OR scrip_group LIKE '%NIFTY_50%' OR scrip_group = 'NIFTY50')
                AND scrip_id NOT IN ('NIFTY50', 'Nifty_50', 'Nifty5')
                AND scrip_id NOT IN (SELECT unnest(%s::text[]))
            """, (self.excluded_stocks,))
            
            stocks = cursor.fetchall()
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(stocks)} Nifty50 stocks to fetch Yahoo fundamentals for")
            
            if not stocks:
                return []
            
            return self.fetch_fundamentals_for_stocks(stocks, batch_size, delay_between_batches)
            
        except Exception as e:
            logger.error(f"Error getting Nifty50 stocks for Yahoo fundamentals fetch: {e}")
            return []
    
    def fetch_fundamentals_for_missing_stocks(self, scrip_ids: Optional[List[str]] = None, 
                                               fetch_holdings: bool = True, fetch_nifty50: bool = True,
                                               batch_size: int = 5, delay_between_batches: float = 3.45) -> Dict:
        """
        Fetch fundamental data from Yahoo Finance for stocks
        
        Args:
            scrip_ids: Optional list of specific stock symbols to fetch
            fetch_holdings: Whether to fetch for holdings
            fetch_nifty50: Whether to fetch for Nifty50 stocks
            batch_size: Number of stocks to process in each batch
            delay_between_batches: Delay in seconds between batches
            
        Returns:
            Dictionary with success status and statistics
        """
        try:
            all_results = []
            
            if scrip_ids:
                # Fetch for specific stocks
                conn = get_db_connection()
                cursor = conn.cursor()
                
                placeholders = ','.join(['%s'] * len(scrip_ids))
                cursor.execute(f"""
                    SELECT scrip_id, COALESCE(yahoo_code, scrip_id || '.NS') as yahoo_code
                    FROM my_schema.master_scrips 
                    WHERE scrip_id IN ({placeholders})
                    AND scrip_country = 'IN'
                    AND scrip_id NOT IN (SELECT unnest(%s::text[]))
                """, tuple(scrip_ids) + (self.excluded_stocks,))
                
                stocks = cursor.fetchall()
                cursor.close()
                conn.close()
                
                logger.info(f"Fetching Yahoo fundamentals for {len(stocks)} specific stocks")
                results = self.fetch_fundamentals_for_stocks(stocks, batch_size, delay_between_batches)
                all_results.extend(results)
            else:
                # Fetch based on flags
                if fetch_holdings:
                    logger.info("Fetching Yahoo fundamentals for holdings...")
                    holdings_results = self.fetch_fundamentals_for_holdings(batch_size, delay_between_batches)
                    all_results.extend(holdings_results)
                
                if fetch_nifty50:
                    logger.info("Fetching Yahoo fundamentals for Nifty50 stocks...")
                    nifty50_results = self.fetch_fundamentals_for_nifty50(batch_size, delay_between_batches)
                    all_results.extend(nifty50_results)
            
            success_count = sum(1 for r in all_results if r.get('status') == 'success')
            failed_count = sum(1 for r in all_results if r.get('status') == 'failed')
            cancelled_count = sum(1 for r in all_results if r.get('status') == 'cancelled')
            
            logger.info(f"Yahoo fundamentals fetch completed: {success_count} succeeded, {failed_count} failed, {cancelled_count} cancelled")
            return {
                'success': True,
                'succeeded': success_count,
                'failed': failed_count,
                'cancelled': cancelled_count,
                'total': len(all_results),
                'results': all_results
            }
            
        except Exception as e:
            logger.error(f"Error in fetch_fundamentals_for_missing_stocks: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }


# Legacy function for backward compatibility
def fetch_fundamentals_for_missing_stocks():
    """
    Fetch fundamental data (actions, income statements, insider transactions) 
    from Yahoo Finance for stocks that don't have this data yet
    """
    fetcher = YahooFundamentalsFetcher()
    return fetcher.fetch_fundamentals_for_missing_stocks()


# Main execution
if __name__ == "__main__":
    fetcher = YahooFundamentalsFetcher()
    result = fetcher.fetch_fundamentals_for_missing_stocks()
    print(f"Fetch result: {result}")
