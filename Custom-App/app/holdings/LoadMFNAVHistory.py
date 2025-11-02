"""
One-time script to load historical NAV data for all Mutual Funds in holdings
Fetches at least 1 year (365 days) of historical NAV data
"""
import sys
import os
import logging
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holdings.MFNAVFetcher import MFNAVFetcher
from common.Boilerplate import get_db_connection
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_all_mf_nav_history(days: int = 365):
    """
    Load historical NAV data for all MFs in holdings
    
    Args:
        days: Number of days of historical data to fetch (default: 365)
    """
    logging.info(f"Starting MF NAV history load for {days} days")
    
    fetcher = MFNAVFetcher()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all unique MFs from holdings
        cursor.execute("""
            SELECT DISTINCT tradingsymbol, fund
            FROM my_schema.mf_holdings
            WHERE tradingsymbol IS NOT NULL AND tradingsymbol != ''
            ORDER BY tradingsymbol
        """)
        
        mf_list = cursor.fetchall()
        cursor.close()
        conn.close()
        
        logging.info(f"Found {len(mf_list)} unique mutual funds in holdings")
        
        if not mf_list:
            logging.warning("No mutual funds found in holdings")
            return
        
        total_processed = 0
        total_records = 0
        errors = []
        
        for mf_symbol, fund_name in mf_list:
            try:
                logging.info(f"Processing MF: {mf_symbol} ({fund_name})")
                
                # Check if we already have sufficient data
                conn_check = get_db_connection()
                cursor_check = conn_check.cursor()
                
                cursor_check.execute("""
                    SELECT COUNT(*), MIN(nav_date), MAX(nav_date)
                    FROM my_schema.mf_nav_history
                    WHERE mf_symbol = %s
                """, (mf_symbol,))
                
                count, min_date, max_date = cursor_check.fetchone()
                cursor_check.close()
                conn_check.close()
                
                # Check if we have enough recent data
                today = datetime.now().date()
                if count and max_date:
                    days_old = (today - max_date).days
                    if days_old <= 1 and count >= days:
                        logging.info(f"Skipping {mf_symbol} - already has {count} records with latest data from {max_date}")
                        continue
                
                # Fetch NAV data
                result = fetcher.fetch_and_save_nav(
                    mf_symbol=mf_symbol,
                    fund_name=fund_name,
                    days=days
                )
                
                if result['success']:
                    total_processed += 1
                    total_records += result['records']
                    logging.info(f"✓ {mf_symbol}: {result['records']} records from {result['source']}")
                else:
                    errors.append(f"{mf_symbol}: {result.get('error', result.get('message', 'Unknown error'))}")
                    logging.warning(f"✗ {mf_symbol}: {result.get('error', result.get('message', 'Failed'))}")
                
                # Rate limiting - wait between requests to avoid overwhelming APIs
                time.sleep(2)  # 2 seconds between requests
                
            except Exception as e:
                error_msg = f"{mf_symbol}: {str(e)}"
                errors.append(error_msg)
                logging.error(f"Error processing {mf_symbol}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        # Summary
        logging.info("=" * 60)
        logging.info(f"MF NAV History Load Complete")
        logging.info(f"Total MFs processed: {total_processed}")
        logging.info(f"Total records inserted: {total_records}")
        logging.info(f"Errors: {len(errors)}")
        
        if errors:
            logging.warning("Errors encountered:")
            for error in errors[:10]:  # Show first 10 errors
                logging.warning(f"  - {error}")
            if len(errors) > 10:
                logging.warning(f"  ... and {len(errors) - 10} more errors")
        
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"Error in load_all_mf_nav_history: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load historical NAV data for all MFs')
    parser.add_argument('--days', type=int, default=365, help='Number of days of historical data (default: 365)')
    
    args = parser.parse_args()
    
    try:
        load_all_mf_nav_history(days=args.days)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

