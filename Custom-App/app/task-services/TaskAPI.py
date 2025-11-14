"""
FastAPI application for background tasks container
Handles on-demand triggers for background tasks
"""

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import date
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Background Tasks API", version="1.0.0")

# Add CORS middleware to allow requests from the frontend container
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can be restricted to specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],   # Allow all headers
)

@app.get("/")
async def root():
    return {"message": "Background Tasks API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

def run_insert_ohlc_background():
    """
    Background task function to refresh stock prices asynchronously
    """
    try:
        from kite.InsertOHLC import refresh_stock_prices
        logger.info("[Background] Starting OHLC stock price refresh...")
        result = refresh_stock_prices()
        logger.info(f"[Background] OHLC refresh completed: {result.get('message', 'Unknown status')}")
        logger.info(f"[Background] Stocks processed: {result.get('stocks_processed', 0)}, Records inserted: {result.get('records_inserted', 0)}")
        if result.get('errors'):
            logger.warning(f"[Background] OHLC refresh had {len(result.get('errors', []))} errors")
    except Exception as e:
        logger.error(f"[Background] Error in OHLC refresh: {e}")
        import traceback
        logger.error(traceback.format_exc())

@app.post("/api/insert_ohlc")
async def api_insert_ohlc(background_tasks: BackgroundTasks):
    """API endpoint to insert OHLC price data (runs asynchronously)"""
    try:
        # Add the task to background tasks
        background_tasks.add_task(run_insert_ohlc_background)
        
        logger.info("OHLC refresh task queued for background execution")
        
        return {
            "success": True,
            "message": "OHLC data refresh started in background. Check application logs for progress and completion status.",
            "status": "queued",
            "note": "The refresh is running asynchronously. Data will be updated in the database as it is processed."
        }
    except Exception as e:
        logger.error(f"Error queuing OHLC refresh: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/api/refresh_stock_prices")
async def api_refresh_stock_prices(background_tasks: BackgroundTasks):
    """API endpoint to refresh stock prices (same as insert_ohlc, runs asynchronously)"""
    try:
        # Add the task to background tasks
        background_tasks.add_task(run_insert_ohlc_background)
        
        logger.info("Stock price refresh task queued for background execution")
        
        return {
            "success": True,
            "message": "Stock price refresh started in background. Check application logs for progress and completion status.",
            "status": "queued",
            "note": "The refresh is running asynchronously. Data will be updated in the database as it is processed."
        }
    except Exception as e:
        logger.error(f"Error queuing stock price refresh: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

def run_prophet_predictions_background(
    prediction_days: int,
    limit: Optional[int],
    fetch_sentiment: bool,
    run_date: date
):
    """
    Background task function to generate Prophet predictions asynchronously
    
    Args:
        prediction_days: Number of days to predict ahead
        limit: Optional limit on number of stocks to process
        fetch_sentiment: Whether to fetch and calculate sentiment
        run_date: Date for the predictions
    """
    try:
        from stocks.ProphetPricePredictor import ProphetPricePredictor
        
        # Generate predictions with sentiment integration
        predictor = ProphetPricePredictor(prediction_days=prediction_days, enable_sentiment=fetch_sentiment)
        
        # Generate predictions and save immediately after each calculation
        logger.info(f"[Background] Starting Prophet prediction generation for run_date={run_date}, prediction_days={prediction_days}, limit={limit}")
        predictions = predictor.predict_all_stocks(limit=limit, prediction_days=prediction_days, save_immediately=True, run_date=run_date)
        
        logger.info(f"[Background] Prophet prediction generation completed: {len(predictions)} predictions generated and saved")
        
        if not predictions:
            error_msg = "No predictions generated. This could be due to: insufficient data (need at least 60 days), Prophet model errors, or data quality issues. Check application logs for details."
            logger.warning(f"[Background] {error_msg}")
            return
        
        logger.info(f"[Background] Successfully generated and saved {len(predictions)} predictions for {prediction_days} days")
        
    except Exception as e:
        logger.error(f"[Background] Error generating Prophet predictions: {e}")
        import traceback
        logger.error(traceback.format_exc())

@app.get("/api/prophet_predictions/generate")
async def api_generate_prophet_predictions(
    background_tasks: BackgroundTasks,
    prediction_days: int = Query(30, description="Number of days to predict ahead (default: 30, supports 30, 60, 90, 180, etc.)"),
    limit: int = Query(None, description="Limit number of stocks to process (default: all)"),
    fetch_fundamentals: bool = Query(True, description="Check and fetch fundamental data if >30 days old (monthly refresh)"),
    force_fundamentals: bool = Query(False, description="Force fetch fundamentals even if recent data exists"),
    fetch_sentiment: bool = Query(True, description="Fetch and calculate sentiment (always daily)")
):
    """API endpoint to generate Prophet price predictions for all stocks with sentiment analysis (runs asynchronously)"""
    try:
        from datetime import date
        
        run_date = date.today()
        
        # Add the prediction generation task to background tasks
        background_tasks.add_task(
            run_prophet_predictions_background,
            prediction_days=prediction_days,
            limit=limit,
            fetch_sentiment=fetch_sentiment,
            run_date=run_date
        )
        
        logger.info(f"Prophet prediction generation task queued for background execution: run_date={run_date}, prediction_days={prediction_days}, limit={limit}")
        
        return {
            "success": True,
            "message": f"Prophet prediction generation started in background for {prediction_days} days. Check application logs for progress and completion status.",
            "run_date": str(run_date),
            "prediction_days": prediction_days,
            "limit": limit,
            "fetch_sentiment": fetch_sentiment,
            "status": "queued",
            "note": "The prediction generation is running asynchronously. Predictions will be saved to the database as they are calculated. Monitor application logs for progress."
        }
        
    except Exception as e:
        logger.error(f"Error queuing Prophet prediction generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "run_date": str(run_date) if 'run_date' in locals() else None
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)