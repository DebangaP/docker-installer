"""
Background task for calculating Supertrend values for all holdings.
This runs asynchronously to avoid blocking the dashboard.
"""
import logging
from typing import List, Optional
from datetime import date
from common.Boilerplate import get_db_connection
import psycopg2.extras
from api.utils.technical_indicators import get_latest_supertrend
from api.utils.supertrend_cache import set_cached_supertrend

def calculate_supertrend_for_holdings(symbols: Optional[List[str]] = None):
    """
    Background task to calculate supertrend values for holdings.
    
    Args:
        symbols: Optional list of specific symbols to calculate. If None, calculates for all holdings.
    """
    try:
        logging.info(f"[Background] Starting Supertrend calculation for holdings")
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get all holdings if symbols not specified
        if symbols is None:
            cursor.execute("""
                SELECT DISTINCT trading_symbol
                FROM my_schema.holdings
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                AND trading_symbol IS NOT NULL
            """)
            rows = cursor.fetchall()
            symbols = [row['trading_symbol'] for row in rows if row['trading_symbol']]
        
        cursor.close()
        conn.close()
        
        if not symbols:
            logging.warning("[Background] No symbols found for Supertrend calculation")
            return
        
        logging.info(f"[Background] Calculating Supertrend for {len(symbols)} symbols")
        
        calculated_count = 0
        error_count = 0
        
        for symbol in symbols:
            try:
                # Calculate supertrend (force_recalculate=False means it will use DB if exists, otherwise calculate and store)
                # This ensures we only calculate once per day per symbol
                result = get_latest_supertrend(symbol, conn=None, force_recalculate=False)
                
                if result is not None:
                    # Cache the result for faster access
                    set_cached_supertrend(symbol, result)
                    calculated_count += 1
                    
                    if calculated_count % 10 == 0:
                        logging.info(f"[Background] Processed Supertrend for {calculated_count}/{len(symbols)} symbols")
                else:
                    error_count += 1
                    logging.debug(f"[Background] No Supertrend result for {symbol}")
                    
            except Exception as e:
                error_count += 1
                logging.warning(f"[Background] Error calculating Supertrend for {symbol}: {e}")
                continue
        
        logging.info(f"[Background] Supertrend calculation completed: {calculated_count} successful, {error_count} errors out of {len(symbols)} total")
        
    except Exception as e:
        logging.error(f"[Background] Error in Supertrend calculation task: {e}")
        import traceback
        logging.error(traceback.format_exc())

