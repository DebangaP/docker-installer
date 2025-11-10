"""
Scheduled Job for Accumulation/Distribution Analysis
Runs daily after market close to analyze stocks for accumulation/distribution patterns
"""

import logging
from datetime import date
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.AccumulationDistributionAnalyzer import AccumulationDistributionAnalyzer
from common.Boilerplate import get_db_connection
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_stocks_list() -> list:
    """
    Get list of all stocks to analyze
    
    Returns:
        List of stock symbols
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get all stocks from rt_intraday_price that have recent data
        cursor.execute("""
            SELECT DISTINCT scrip_id
            FROM my_schema.rt_intraday_price
            WHERE country = 'IN'
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
            AND price_date::date >= CURRENT_DATE - make_interval(days => 30)
            ORDER BY scrip_id
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        stocks = [row['scrip_id'] for row in rows]
        logger.info(f"Found {len(stocks)} stocks to analyze")
        return stocks
        
    except Exception as e:
        logger.error(f"Error getting stocks list: {e}")
        return []


def run_accumulation_distribution_analysis():
    """
    Run accumulation/distribution analysis for all stocks
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting Accumulation/Distribution Analysis Job")
        logger.info(f"Date: {date.today()}")
        logger.info("=" * 80)
        
        analyzer = AccumulationDistributionAnalyzer()
        analysis_date = date.today()
        
        # Get list of stocks to analyze
        stocks = get_stocks_list()
        
        if not stocks:
            logger.warning("No stocks found to analyze")
            return {"success": False, "error": "No stocks found"}
        
        # Analyze each stock
        results = {
            'total_stocks': len(stocks),
            'analyzed': 0,
            'success': 0,
            'failed': 0,
            'accumulation': 0,
            'distribution': 0,
            'neutral': 0,
            'errors': []
        }
        
        for i, scrip_id in enumerate(stocks, 1):
            try:
                logger.info(f"[{i}/{len(stocks)}] Analyzing {scrip_id}...")
                
                # Analyze stock
                result = analyzer.analyze_stock(scrip_id, lookback_days=30)
                
                if result:
                    # Save result to database
                    success = analyzer.save_analysis_result(scrip_id, analysis_date, result)
                    
                    if success:
                        results['analyzed'] += 1
                        results['success'] += 1
                        
                        state = result.get('state', 'NEUTRAL')
                        if state == 'ACCUMULATION':
                            results['accumulation'] += 1
                        elif state == 'DISTRIBUTION':
                            results['distribution'] += 1
                        else:
                            results['neutral'] += 1
                        
                        logger.debug(f"{scrip_id}: {state} (confidence: {result.get('confidence_score', 0):.1f}%)")
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"{scrip_id}: Failed to save result")
                        logger.warning(f"Failed to save result for {scrip_id}")
                else:
                    results['failed'] += 1
                    results['errors'].append(f"{scrip_id}: No analysis result")
                    logger.debug(f"No analysis result for {scrip_id} (insufficient data)")
                    
            except Exception as e:
                results['failed'] += 1
                error_msg = f"{scrip_id}: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(f"Error analyzing {scrip_id}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Log summary
        logger.info("=" * 80)
        logger.info("Analysis Summary")
        logger.info("=" * 80)
        logger.info(f"Total stocks: {results['total_stocks']}")
        logger.info(f"Analyzed: {results['analyzed']}")
        logger.info(f"Success: {results['success']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Accumulation: {results['accumulation']}")
        logger.info(f"Distribution: {results['distribution']}")
        logger.info(f"Neutral: {results['neutral']}")
        
        if results['errors']:
            logger.warning(f"Errors encountered: {len(results['errors'])}")
            if len(results['errors']) <= 10:
                for error in results['errors']:
                    logger.warning(f"  - {error}")
            else:
                for error in results['errors'][:10]:
                    logger.warning(f"  - {error}")
                logger.warning(f"  ... and {len(results['errors']) - 10} more errors")
        
        logger.info("=" * 80)
        
        return {
            "success": True,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in accumulation/distribution analysis job: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = run_accumulation_distribution_analysis()
    if result.get('success'):
        sys.exit(0)
    else:
        sys.exit(1)

