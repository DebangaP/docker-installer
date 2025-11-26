"""
TPO Predictions Router
API endpoints for TPO-based support and resistance predictions
"""

from fastapi import APIRouter, Query
from typing import Optional
import logging
from datetime import datetime
from market.CalculateTPO import PostgresDataFetcher, TPOProfile

router = APIRouter(prefix="/api/tpo-predictions", tags=["tpo-predictions"])
logger = logging.getLogger(__name__)

# ANALYSIS_DATE will be accessed via query parameter or current date
# Avoid circular import by not importing from KiteAccessToken directly
ANALYSIS_DATE = None


@router.get("/support-resistance")
async def get_support_resistance_predictions(
    analysis_date: Optional[str] = Query(None, description="Analysis date in YYYY-MM-DD format")
):
    """
    Get predicted support and resistance levels based on TPO analysis
    
    Uses TPO Cluster Analysis, Value Area Extensions, and Fibonacci-like extensions
    to predict upcoming support and resistance levels for both pre-market and live market periods.
    """
    try:
        DB_CONFIG = {
            'host': 'postgres',
            'database': 'mydb',
            'user': 'postgres',
            'password': 'postgres',
            'port': 5432
        }
        
        db_fetcher = PostgresDataFetcher(**DB_CONFIG)
        
        # Determine analysis date
        if analysis_date:
            target_date = analysis_date
        elif ANALYSIS_DATE:
            target_date = ANALYSIS_DATE
        else:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get pre-market TPO data
        pre_market_df = db_fetcher.fetch_tick_data(
            table_name='ticks',
            instrument_token=256265,
            start_time=f'{target_date} 09:05:00.000 +0530',
            end_time=f'{target_date} 09:15:00.000 +0530'
        )
        
        # Get real-time/live market TPO data
        if analysis_date or ANALYSIS_DATE:
            end_time = f'{target_date} 15:30:00.000 +0530'
        else:
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.000 +0530")
        
        real_time_df = db_fetcher.fetch_tick_data(
            table_name='ticks',
            instrument_token=256265,
            start_time=f'{target_date} 09:15:00.000 +0530',
            end_time=end_time
        )
        
        # Get current price from latest tick data
        current_price = None
        if not real_time_df.empty:
            current_price = float(real_time_df['last_price'].iloc[-1])
        elif not pre_market_df.empty:
            current_price = float(pre_market_df['last_price'].iloc[-1])
        
        # Calculate predictions for pre-market
        pre_market_predictions = {
            'support_levels': [],
            'resistance_levels': [],
            'current_price': current_price,
            'poc': None,
            'vah': None,
            'val': None
        }
        
        if not pre_market_df.empty:
            pre_market_tpo = TPOProfile(tick_size=5)
            pre_market_tpo.calculate_tpo(pre_market_df)
            pre_market_predictions = pre_market_tpo.predict_support_resistance_levels(
                current_price=current_price
            )
        
        # Calculate predictions for live market
        live_market_predictions = {
            'support_levels': [],
            'resistance_levels': [],
            'current_price': current_price,
            'poc': None,
            'vah': None,
            'val': None
        }
        
        if not real_time_df.empty:
            live_market_tpo = TPOProfile(tick_size=5)
            live_market_tpo.calculate_tpo(real_time_df)
            live_market_predictions = live_market_tpo.predict_support_resistance_levels(
                current_price=current_price
            )
        
        return {
            "success": True,
            "analysis_date": target_date,
            "is_historical": bool(analysis_date or ANALYSIS_DATE),
            "pre_market": pre_market_predictions,
            "live_market": live_market_predictions
        }
        
    except Exception as e:
        logger.error(f"Error generating TPO predictions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "analysis_date": target_date if 'target_date' in locals() else None,
            "pre_market": {
                'support_levels': [],
                'resistance_levels': [],
                'current_price': None,
                'poc': None,
                'vah': None,
                'val': None
            },
            "live_market": {
                'support_levels': [],
                'resistance_levels': [],
                'current_price': None,
                'poc': None,
                'vah': None,
                'val': None
            }
        }

