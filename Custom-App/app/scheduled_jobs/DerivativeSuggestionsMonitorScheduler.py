"""
Scheduled Job for Monitoring Derivative Suggestions
Runs periodically during market hours to check for exit conditions and calculate P&L
"""

import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.services.derivative_suggestions_monitor import monitor_derivative_suggestions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_monitoring_job():
    """
    Run monitoring job to check for exit conditions on MOCKED suggestions
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting Derivative Suggestions Monitoring Job")
        logger.info("=" * 80)
        
        result = monitor_derivative_suggestions()
        
        if 'error' in result:
            logger.error(f"Monitoring job failed: {result['error']}")
            return False
        
        logger.info("Monitoring job completed successfully")
        logger.info(f"Statistics: {result}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in monitoring job: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_monitoring_job()
    sys.exit(0 if success else 1)

