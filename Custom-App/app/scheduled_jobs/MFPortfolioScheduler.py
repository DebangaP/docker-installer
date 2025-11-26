"""
Scheduled Job for Mutual Fund Portfolio Holdings Update
Runs monthly to fetch and update MF portfolio constituents
"""

import logging
from datetime import date, datetime
from typing import Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holdings.MFPortfolioFetcher import MFPortfolioFetcher
from common.Boilerplate import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_monthly_mf_portfolio_job(force_refresh: bool = False):
    """
    Run monthly MF portfolio holdings update job
    
    Args:
        force_refresh: Force refresh even if recent data exists (default: False)
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting Monthly MF Portfolio Holdings Update Job")
        logger.info(f"Date: {date.today()}")
        logger.info(f"Force Refresh: {force_refresh}")
        logger.info("=" * 80)
        
        fetcher = MFPortfolioFetcher()
        
        # Check if we need to update (MF portfolios typically update monthly)
        # We'll update if last portfolio date is more than 25 days old
        if not force_refresh:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT MAX(portfolio_date) as latest_date
                FROM my_schema.mf_portfolio_holdings
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0]:
                latest_date = result[0]
                days_since_update = (date.today() - latest_date).days
                
                if days_since_update < 25:
                    logger.info(f"Portfolio data is recent (last update: {latest_date}, {days_since_update} days ago)")
                    logger.info("Skipping update. Use --force-refresh to update anyway.")
                    logger.info("=" * 80)
                    logger.info("Monthly MF Portfolio Holdings Update Job Skipped (recent data)")
                    logger.info("=" * 80)
                    return
        
        # Fetch portfolios for all held MFs
        logger.info("Fetching portfolio holdings for all held mutual funds...")
        results = fetcher.fetch_portfolios_for_all_held_mfs()
        
        total_mfs = results.get('total_mfs', 0)
        successful = results.get('successful', 0)
        failed = results.get('failed', 0)
        
        logger.info(f"Portfolio fetch completed:")
        logger.info(f"  Total MFs: {total_mfs}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        
        # Log details for failed MFs
        if failed > 0:
            logger.warning("Failed MF portfolio fetches:")
            for result in results.get('results', []):
                if not result.get('success'):
                    logger.warning(f"  - {result.get('mf_symbol')}: {result.get('error', 'Unknown error')}")
        
        logger.info("=" * 80)
        logger.info("Monthly MF Portfolio Holdings Update Job Completed")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error in monthly MF portfolio job: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Run the job
    # Can be called from command line or scheduled via cron
    import argparse
    
    parser = argparse.ArgumentParser(description='Run monthly MF portfolio holdings update job')
    parser.add_argument('--force-refresh', action='store_true', 
                       help='Force refresh even if recent data exists')
    
    args = parser.parse_args()
    
    run_monthly_mf_portfolio_job(force_refresh=args.force_refresh)

