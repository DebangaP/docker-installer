"""
Scheduled Job for Automated Prediction Generation
Runs daily to fetch fundamental data, calculate sentiment, and generate predictions
"""

import logging
from datetime import date
from typing import Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fundamentals.ScreenerDataFetcher import ScreenerDataFetcher
from sentiment.NewsSentimentAnalyzer import NewsSentimentAnalyzer
from sentiment.FundamentalSentimentAnalyzer import FundamentalSentimentAnalyzer
from sentiment.CombinedSentimentCalculator import CombinedSentimentCalculator
from stocks.ProphetPricePredictor import ProphetPricePredictor
from common.Boilerplate import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_daily_prediction_job(
    prediction_days: int = 30,
    fetch_fundamentals: bool = True,
    force_fundamentals: bool = False,
    fetch_sentiment: bool = True,
    limit: Optional[int] = None
):
    """
    Run daily prediction job
    
    Args:
        prediction_days: Number of days to predict ahead (default: 30)
        fetch_fundamentals: Whether to fetch fundamental data (default: True, but only if >30 days old)
        force_fundamentals: Force fetch fundamentals even if recent data exists (default: False)
        fetch_sentiment: Whether to fetch and calculate sentiment (default: True, always runs daily)
        limit: Optional limit on number of stocks to process
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting Daily Prediction Job")
        logger.info(f"Date: {date.today()}")
        logger.info(f"Prediction Days: {prediction_days}")
        logger.info(f"Fetch Fundamentals: {fetch_fundamentals} (force: {force_fundamentals})")
        logger.info(f"Fetch Sentiment: {fetch_sentiment} (always daily)")
        logger.info("=" * 80)
        
        # Step 1: Fetch fundamental data (only if >30 days old, unless forced)
        # Fundamental data doesn't change frequently, so we fetch monthly
        if fetch_fundamentals:
            try:
                logger.info("Step 1: Checking and fetching fundamental data (monthly refresh)...")
                fetcher = ScreenerDataFetcher()
                
                # Fetch for holdings (only if data is >30 days old) - in batches
                logger.info("Checking fundamentals for holdings (will skip if data is <30 days old, batch processing)...")
                holdings_results = fetcher.fetch_all_holdings_fundamentals(
                    force_refresh=force_fundamentals,
                    days_threshold=30,
                    batch_size=5,  # Process 5 stocks at a time
                    delay_between_batches=0.0  # No delay between batches
                )
                holdings_success = sum(1 for r in holdings_results if r.get('status') == 'success')
                holdings_skipped = sum(1 for r in holdings_results if r.get('status') == 'skipped')
                logger.info(f"Holdings: {holdings_success} fetched, {holdings_skipped} skipped (recent data), {len(holdings_results) - holdings_success - holdings_skipped} failed")
                
                # Fetch for Nifty50 (only if data is >30 days old) - in batches
                logger.info("Checking fundamentals for Nifty50 stocks (will skip if data is <30 days old, batch processing)...")
                nifty50_results = fetcher.fetch_nifty50_fundamentals(
                    force_refresh=force_fundamentals,
                    days_threshold=30,
                    batch_size=5,  # Process 5 stocks at a time
                    delay_between_batches=0.0  # No delay between batches
                )
                nifty50_success = sum(1 for r in nifty50_results if r.get('status') == 'success')
                nifty50_skipped = sum(1 for r in nifty50_results if r.get('status') == 'skipped')
                logger.info(f"Nifty50: {nifty50_success} fetched, {nifty50_skipped} skipped (recent data), {len(nifty50_results) - nifty50_success - nifty50_skipped} failed")
                
                logger.info("Step 1 completed: Fundamental data check/fetch (monthly refresh)")
            except Exception as e:
                logger.error(f"Error in Step 1 (fundamental data fetch): {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Step 2: Fetch and calculate sentiment (always runs daily)
        # Sentiment changes daily, so we always fetch and calculate
        if fetch_sentiment:
            try:
                logger.info("Step 2: Fetching and calculating sentiment (daily refresh)...")
                
                news_analyzer = NewsSentimentAnalyzer()
                fundamental_analyzer = FundamentalSentimentAnalyzer()
                combined_calculator = CombinedSentimentCalculator()
                
                # Get all stocks
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT scrip_id
                    FROM my_schema.rt_intraday_price
                    WHERE country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                """)
                all_stocks = [row[0] for row in cursor.fetchall()]
                cursor.close()
                conn.close()
                
                if limit:
                    all_stocks = all_stocks[:limit]
                
                logger.info(f"Processing sentiment for {len(all_stocks)} stocks (daily refresh)...")
                
                processed_count = 0
                for idx, stock in enumerate(all_stocks, 1):
                    try:
                        if idx % 10 == 0:
                            logger.info(f"Sentiment progress: {idx}/{len(all_stocks)} stocks processed")
                        
                        # News sentiment (always fetch daily)
                        news_analyzer.fetch_and_analyze(stock, days=7)
                        
                        # Fundamental sentiment (uses existing fundamental data, doesn't fetch)
                        fundamental_analyzer.calculate_fundamental_sentiment(stock)
                        
                        # Combined sentiment (combines news and fundamental sentiment)
                        combined_calculator.calculate_combined_sentiment(stock)
                        
                        processed_count += 1
                    except Exception as e:
                        logger.warning(f"Error processing sentiment for {stock}: {e}")
                        continue
                
                logger.info(f"Step 2 completed: Sentiment analysis for {processed_count}/{len(all_stocks)} stocks (daily refresh)")
            except Exception as e:
                logger.error(f"Error in Step 2 (sentiment analysis): {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Step 3: Generate predictions
        try:
            logger.info("Step 3: Generating Prophet predictions...")
            
            predictor = ProphetPricePredictor(
                prediction_days=prediction_days,
                enable_sentiment=fetch_sentiment
            )
            
            run_date = date.today()
            predictions = predictor.predict_all_stocks(
                limit=limit,
                prediction_days=prediction_days,
                save_immediately=True,
                run_date=run_date
            )
            
            logger.info(f"Step 3 completed: Generated {len(predictions)} predictions")
        except Exception as e:
            logger.error(f"Error in Step 3 (prediction generation): {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info("=" * 80)
        logger.info("Daily Prediction Job Completed")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error in daily prediction job: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Run the job
    # Can be called from command line or scheduled via cron
    import argparse
    
    parser = argparse.ArgumentParser(description='Run daily prediction job')
    parser.add_argument('--prediction-days', type=int, default=30, help='Number of days to predict ahead')
    parser.add_argument('--no-fundamentals', action='store_true', help='Skip fundamental data fetch')
    parser.add_argument('--force-fundamentals', action='store_true', help='Force fetch fundamentals even if recent data exists')
    parser.add_argument('--no-sentiment', action='store_true', help='Skip sentiment analysis')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of stocks to process')
    
    args = parser.parse_args()
    
    run_daily_prediction_job(
        prediction_days=args.prediction_days,
        fetch_fundamentals=not args.no_fundamentals,
        force_fundamentals=args.force_fundamentals,
        fetch_sentiment=not args.no_sentiment,
        limit=args.limit
    )

