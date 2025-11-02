"""
Daily script to refresh MF NAV data
Similar to RefreshHoldings.py, but for MF NAV data
Fetches latest NAV data for all MFs in holdings
"""
import sys
import os
import logging
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holdings.MFNAVFetcher import MFNAVFetcher
from common.Boilerplate import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def refresh_mf_nav():
    """
    Refresh NAV data for all MFs in holdings
    Fetches missing or outdated NAV data
    """
    logging.info("Starting MF NAV refresh")
    
    fetcher = MFNAVFetcher()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    result = {
        'success': False,
        'message': '',
        'mfs_processed': 0,
        'records_inserted': 0,
        'errors': []
    }
    
    try:
        # Get all unique MFs from holdings
        cursor.execute("""
            SELECT DISTINCT tradingsymbol, fund
            FROM my_schema.mf_holdings
            WHERE tradingsymbol IS NOT NULL AND tradingsymbol != ''
            ORDER BY tradingsymbol
        """)
        
        mf_list = cursor.fetchall()
        
        if not mf_list:
            logging.warning("No mutual funds found in holdings")
            result['success'] = True
            result['message'] = 'No mutual funds found in holdings'
            return result
        
        logging.info(f"Found {len(mf_list)} unique mutual funds")
        
        today = datetime.now().date()
        mfs_processed = 0
        records_inserted = 0
        errors = []
        
        for mf_symbol, fund_name in mf_list:
            try:
                # Check last NAV date for this MF
                cursor.execute("""
                    SELECT MAX(nav_date) as last_nav_date
                    FROM my_schema.mf_nav_history
                    WHERE mf_symbol = %s
                """, (mf_symbol,))
                
                last_nav = cursor.fetchone()
                last_nav_date = last_nav[0] if last_nav and last_nav[0] else None
                
                # Calculate date range
                if last_nav_date:
                    # Fetch from last NAV date to today
                    days_missing = (today - last_nav_date).days
                    if days_missing <= 1:
                        # Already up to date (within 1 day)
                        logging.debug(f"Skipping {mf_symbol} - already up to date (last NAV: {last_nav_date})")
                        continue
                    start_date = last_nav_date + timedelta(days=1)
                    days_to_fetch = days_missing
                else:
                    # No historical data, fetch last 365 days
                    start_date = today - timedelta(days=365)
                    days_to_fetch = 365
                
                logging.info(f"Fetching NAV for {mf_symbol} from {start_date} to {today} ({days_to_fetch} days)")
                
                # Fetch NAV data
                fetch_result = fetcher.fetch_and_save_nav(
                    mf_symbol=mf_symbol,
                    fund_name=fund_name,
                    start_date=start_date,
                    end_date=today,
                    days=days_to_fetch
                )
                
                if fetch_result['success']:
                    mfs_processed += 1
                    records_inserted += fetch_result['records']
                    logging.info(f"✓ {mf_symbol}: {fetch_result['records']} records from {fetch_result['source']}")
                else:
                    error_msg = f"{mf_symbol}: {fetch_result.get('error', fetch_result.get('message', 'Unknown error'))}"
                    errors.append(error_msg)
                    logging.warning(f"✗ {error_msg}")
                
            except Exception as e:
                error_msg = f"{mf_symbol}: {str(e)}"
                errors.append(error_msg)
                logging.error(f"Error processing {mf_symbol}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        result['success'] = True
        result['mfs_processed'] = mfs_processed
        result['records_inserted'] = records_inserted
        result['errors'] = errors
        
        if errors:
            result['message'] = f"Processed {mfs_processed} MFs, inserted {records_inserted} records. {len(errors)} errors occurred."
        else:
            result['message'] = f"Successfully processed {mfs_processed} MFs and inserted {records_inserted} NAV records."
        
        logging.info(result['message'])
        
    except Exception as e:
        error_msg = f"Error in refresh_mf_nav: {str(e)}"
        logging.error(error_msg)
        import traceback
        logging.error(traceback.format_exc())
        result['errors'].append(error_msg)
        result['message'] = error_msg
        
        if conn:
            cursor.close()
            conn.close()
    
    return result

if __name__ == '__main__':
    try:
        result = refresh_mf_nav()
        if result['success']:
            logging.info("MF NAV refresh completed successfully")
            sys.exit(0)
        else:
            logging.error("MF NAV refresh completed with errors")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

