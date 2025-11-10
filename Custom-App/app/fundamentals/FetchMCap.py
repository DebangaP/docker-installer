"""
Market Cap Fetcher
Fetches market cap data from Yahoo Finance and updates master_scrips table
"""

import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import logging
import time
from typing import List, Dict, Optional
from common.Boilerplate import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketCapFetcher:
    """
    Fetches market cap data from Yahoo Finance and updates master_scrips table
    """
    
    def __init__(self):
        self._cancellation_flag_key = "mcap_fetch_cancelled"
        # Initialize Redis client once for reuse
        self._redis_client = None
        self._init_redis_client()
        # Ensure database table exists
        self._ensure_system_flags_table()
    
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
    
    def update_mcap_for_stocks(self, scrip_ids: List[str], batch_size: int = 10, delay_between_batches: float = 1.0) -> List[Dict]:
        """
        Update market cap for a list of stocks
        
        Args:
            scrip_ids: List of stock symbols to update
            batch_size: Number of stocks to process in each batch
            delay_between_batches: Delay in seconds between batches
            
        Returns:
            List of results with status for each stock
        """
        results = []
        updated_count = 0
        failed_count = 0
        
        # Clear cancellation flag at start
        self.set_cancellation_flag(False)
        
        # Get yahoo codes for all stocks
        conn = get_db_connection()
        cursor = conn.cursor()
        
        placeholders = ','.join(['%s'] * len(scrip_ids))
        cursor.execute(f"""
            SELECT scrip_id, yahoo_code 
            FROM my_schema.master_scrips 
            WHERE scrip_id IN ({placeholders})
            AND yahoo_code IS NOT NULL
            AND scrip_country = 'IN'
        """, tuple(scrip_ids))
        
        stocks = cursor.fetchall()
        cursor.close()
        conn.close()
        
        logger.info(f"Found {len(stocks)} stocks with yahoo_code to update market cap")
        
        # Process in batches
        cancelled = False
        for batch_idx in range(0, len(stocks), batch_size):
            # Check for cancellation before each batch
            if self.check_cancellation():
                logger.info("Market cap update cancelled by user")
                cancelled = True
                break
            
            batch = stocks[batch_idx:batch_idx + batch_size]
            logger.info(f"Processing batch {batch_idx // batch_size + 1} ({len(batch)} stocks)...")
            
            batch_conn = get_db_connection()
            batch_cursor = batch_conn.cursor()
            
            for scrip_id, yahoo_code in batch:
                # Check for cancellation before each stock
                if self.check_cancellation():
                    logger.info("Market cap update cancelled by user")
                    cancelled = True
                    break
                
                try:
                    # Fetch market cap from Yahoo Finance
                    stock = yf.Ticker(yahoo_code)
                    info = stock.info
                    market_cap = info.get('marketCap', None)
                    
                    if market_cap:
                        # Convert from USD to INR crores
                        # Market cap is in USD, convert to INR (approximate rate: 88)
                        # Then convert to crores (divide by 10,000,000)
                        market_cap_inr = market_cap * 88  # USD to INR
                        market_cap_crores = round(market_cap_inr / 10_000_000, 2)
                        
                        # Update market cap in database
                        update_query = """
                            UPDATE my_schema.master_scrips 
                            SET scrip_mcap = %s 
                            WHERE scrip_id = %s
                        """
                        batch_cursor.execute(update_query, (market_cap_crores, scrip_id))
                        updated_count += 1
                        logger.debug(f"Updated market cap for {scrip_id}: {market_cap_crores} Cr")
                        results.append({
                            'scrip_id': scrip_id,
                            'status': 'success',
                            'market_cap': market_cap_crores
                        })
                    else:
                        logger.warning(f"No market cap data available for {scrip_id} ({yahoo_code})")
                        failed_count += 1
                        results.append({
                            'scrip_id': scrip_id,
                            'status': 'failed',
                            'error': 'No market cap data available'
                        })
                        
                except Exception as e:
                    logger.error(f"Error fetching market cap for {scrip_id} ({yahoo_code}): {e}")
                    failed_count += 1
                    results.append({
                        'scrip_id': scrip_id,
                        'status': 'failed',
                        'error': str(e)
                    })
                    continue
                
                # Check for cancellation after each stock
                if self.check_cancellation():
                    logger.info("Market cap update cancelled by user")
                    cancelled = True
                    break
            
            # Commit batch
            batch_conn.commit()
            batch_cursor.close()
            batch_conn.close()
            
            # Check for cancellation after batch
            if self.check_cancellation():
                logger.info("Market cap update cancelled by user")
                cancelled = True
                break
            
            # Delay between batches (if delay > 0)
            if delay_between_batches > 0 and batch_idx + batch_size < len(stocks):
                # Check for cancellation during delay
                delay_elapsed = 0
                while delay_elapsed < delay_between_batches:
                    if self.check_cancellation():
                        logger.info("Market cap update cancelled by user")
                        cancelled = True
                        break
                    time.sleep(0.1)
                    delay_elapsed += 0.1
                if cancelled:
                    break
        
        logger.info(f"Market cap update completed: {updated_count} updated, {failed_count} failed")
        return results
    
    def update_mcap_for_holdings(self, batch_size: int = 10, delay_between_batches: float = 1.0) -> List[Dict]:
        """
        Update market cap for all holdings
        
        Args:
            batch_size: Number of stocks to process in each batch
            delay_between_batches: Delay in seconds between batches
            
        Returns:
            List of results with status for each stock
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all holdings
            cursor.execute("""
                SELECT DISTINCT ms.scrip_id
                FROM my_schema.master_scrips ms
                INNER JOIN my_schema.holdings h ON h.trading_symbol = ms.scrip_id
                WHERE ms.scrip_country = 'IN'
                AND ms.yahoo_code IS NOT NULL
            """)
            
            holdings = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(holdings)} holdings to update market cap")
            
            if not holdings:
                return []
            
            return self.update_mcap_for_stocks(holdings, batch_size, delay_between_batches)
            
        except Exception as e:
            logger.error(f"Error getting holdings for market cap update: {e}")
            return []
    
    def update_mcap_for_nifty50(self, batch_size: int = 10, delay_between_batches: float = 1.0) -> List[Dict]:
        """
        Update market cap for Nifty50 stocks
        
        Args:
            batch_size: Number of stocks to process in each batch
            delay_between_batches: Delay in seconds between batches
            
        Returns:
            List of results with status for each stock
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get Nifty50 stocks (exclude the index itself)
            cursor.execute("""
                SELECT scrip_id
                FROM my_schema.master_scrips
                WHERE scrip_country = 'IN'
                AND yahoo_code IS NOT NULL
                AND (scrip_group LIKE '%NIFTY50%' OR scrip_group LIKE '%NIFTY_50%' OR scrip_group = 'NIFTY50')
                AND scrip_id NOT IN ('NIFTY50', 'Nifty_50', 'Nifty5')
            """)
            
            nifty50_stocks = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(nifty50_stocks)} Nifty50 stocks to update market cap")
            
            if not nifty50_stocks:
                return []
            
            return self.update_mcap_for_stocks(nifty50_stocks, batch_size, delay_between_batches)
            
        except Exception as e:
            logger.error(f"Error getting Nifty50 stocks for market cap update: {e}")
            return []
    
    def update_mcap(self, scrip_ids: Optional[List[str]] = None, fetch_holdings: bool = True, 
                    fetch_nifty50: bool = True, batch_size: int = 10, delay_between_batches: float = 1.0) -> Dict:
        """
        Update market cap for stocks
        
        Args:
            scrip_ids: Optional list of specific stock symbols to update
            fetch_holdings: Whether to update market cap for holdings
            fetch_nifty50: Whether to update market cap for Nifty50 stocks
            batch_size: Number of stocks to process in each batch
            delay_between_batches: Delay in seconds between batches
            
        Returns:
            Dictionary with success status and statistics
        """
        try:
            all_results = []
            
            if scrip_ids:
                # Update specific stocks
                logger.info(f"Updating market cap for {len(scrip_ids)} specific stocks")
                results = self.update_mcap_for_stocks(scrip_ids, batch_size, delay_between_batches)
                all_results.extend(results)
            else:
                # Update based on flags
                if fetch_holdings:
                    logger.info("Updating market cap for holdings...")
                    holdings_results = self.update_mcap_for_holdings(batch_size, delay_between_batches)
                    all_results.extend(holdings_results)
                
                if fetch_nifty50:
                    logger.info("Updating market cap for Nifty50 stocks...")
                    nifty50_results = self.update_mcap_for_nifty50(batch_size, delay_between_batches)
                    all_results.extend(nifty50_results)
            
            success_count = sum(1 for r in all_results if r.get('status') == 'success')
            failed_count = sum(1 for r in all_results if r.get('status') == 'failed')
            
            logger.info(f"Market cap update completed: {success_count} updated, {failed_count} failed")
            return {
                'success': True,
                'updated': success_count,
                'failed': failed_count,
                'total': len(all_results),
                'results': all_results
            }
            
        except Exception as e:
            logger.error(f"Error in update_mcap: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }


# Legacy function for backward compatibility
def update_mcap():
    """
    Update market cap for all Indian stocks in master_scrips table
    """
    fetcher = MarketCapFetcher()
    return fetcher.update_mcap(fetch_holdings=True, fetch_nifty50=True)


def fetch_market_cap(symbols):
    """
    Fetch market cap for a list of symbols and export to CSV
    
    Args:
        symbols: List of stock symbols (without exchange suffix)
    
    Returns:
        DataFrame with market cap data
    """
    results = []
    
    for symbol in symbols:
        market_cap = 'N/A'
        final_symbol = symbol
        
        # Try with .NS suffix first (NSE)
        try:
            stock = yf.Ticker(f"{symbol}.NS")
            info = stock.info
            market_cap = info.get('marketCap', 'N/A')
            
            if market_cap != 'N/A':
                # Convert to crores
                market_cap_inr = market_cap * 83  # USD to INR
                market_cap_crores = round(market_cap_inr / 10_000_000, 2)
                market_cap = market_cap_crores
                final_symbol = f"{symbol}.NS"
            else:
                raise ValueError("No market cap data for .NS")
                
        except Exception:
            # If .NS fails, try with .BO suffix (BSE)
            try:
                stock = yf.Ticker(f"{symbol}.BO")
                info = stock.info
                market_cap = info.get('marketCap', 'N/A')
                
                if market_cap != 'N/A':
                    # Convert to crores
                    market_cap_inr = market_cap * 83  # USD to INR
                    market_cap_crores = round(market_cap_inr / 10_000_000, 2)
                    market_cap = market_cap_crores
                    final_symbol = f"{symbol}.BO"
                else:
                    logger.warning(f"No market cap data for {symbol}.BO")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} (.NS and .BO): {e}")
        
        # Append result
        results.append({
            'Symbol': final_symbol,
            'Market Cap (Crores)': market_cap
        })
    
    # Create DataFrame and export to CSV
    df = pd.DataFrame(results)
    df.to_csv('market_cap_data.csv', index=False)
    logger.info(f"\nData exported to 'market_cap_data.csv'")
    
    return df


# Example usage
if __name__ == "__main__":
    # Update market cap for all stocks in database
    fetcher = MarketCapFetcher()
    result = fetcher.update_mcap()
    print(f"Update result: {result}")
