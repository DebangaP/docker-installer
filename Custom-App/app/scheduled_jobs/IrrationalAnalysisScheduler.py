"""
Scheduled Job for Irrational Analysis
Runs daily after market close to analyze holdings for irrational moves
"""

import logging
from datetime import date
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holdings.IrrationalMoveAnalyzer import IrrationalMoveAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_irrational_analysis():
    """
    Run irrational analysis for all holdings
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting Irrational Analysis Job")
        logger.info(f"Date: {date.today()}")
        logger.info("=" * 80)
        
        analyzer = IrrationalMoveAnalyzer()
        result = analyzer.analyze_all_holdings()
        
        if result.get('success'):
            logger.info(f"Analysis completed successfully")
            logger.info(f"Total holdings: {result.get('total_holdings', 0)}")
            logger.info(f"Analyzed: {result.get('analyzed', 0)}")
            
            # Log summary of recommendations
            results = result.get('results', [])
            strong = [r for r in results if r.get('exit_recommendation') == 'STRONG']
            moderate = [r for r in results if r.get('exit_recommendation') == 'MODERATE']
            weak = [r for r in results if r.get('exit_recommendation') == 'WEAK']
            
            logger.info(f"Exit Recommendations:")
            logger.info(f"  STRONG: {len(strong)}")
            logger.info(f"  MODERATE: {len(moderate)}")
            logger.info(f"  WEAK: {len(weak)}")
            
            if strong:
                logger.warning("STRONG exit recommendations:")
                for item in strong:
                    logger.warning(f"  - {item.get('trading_symbol')}: Score {item.get('irrational_score')}, Reason: {item.get('exit_reason')}")
            
            analyzer.close()
            return result
        else:
            logger.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            analyzer.close()
            return result
            
    except Exception as e:
        logger.error(f"Error in irrational analysis job: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = run_irrational_analysis()
    if result.get('success'):
        sys.exit(0)
    else:
        sys.exit(1)

