from common.Boilerplate import *
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Request, Query, Form, File, UploadFile, Body
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from typing import Optional, List
import os
from datetime import datetime, timedelta
import json

# Helper function to add cache headers based on endpoint refresh frequency
def get_cache_headers(endpoint_path: str) -> dict:
    """
    Returns Cache-Control header based on endpoint refresh frequency
    Based on crontab refresh intervals:
    - Options data: 5 minutes
    - Holdings: 5 minutes  
    - OHLC: 30 minutes
    - Derivatives suggestions: 1 minute
    - System status: 10 seconds
    - Market data (bias, order flow): 5 minutes
    - Gainers/Losers: 1 hour
    """
    # Cache durations in seconds based on refresh frequency
    cache_durations = {
        # Options data - 5 minutes (300 seconds)
        'options_latest': 300,
        'options_scanner': 300,
        'options_chain': 300,
        'options_data': 300,
        
        # Holdings - 5 minutes (300 seconds)
        'holdings': 300,
        'mf_holdings': 300,
        'today_pnl_summary': 300,
        'portfolio_history': 300,
        'holdings/patterns': 300,
        
        # Derivatives - 1 minute (60 seconds)
        'derivatives_suggestions': 60,
        'derivatives_history': 60,
        
        # Market data - 5 minutes (300 seconds)
        'market_bias': 300,
        'futures_order_flow': 300,
        'premarket_analysis': 300,
        'market_dashboard': 300,
        
        # System status - 10 seconds
        'system_status': 10,
        
        # Positions - 5 minutes
        'positions': 300,
        
        # Gainers/Losers - 30 seconds (to match polling interval)
        'gainers': 30,
        'losers': 30,
        'gainers_losers': 30,
        'top_gainers': 30,
        
        # Swing trades - 30 minutes (1800 seconds)
        'swing_trades': 1800,
        'swing_trades_nifty': 1800,
        'swing_trades_history': 1800,
        
        # Portfolio hedge - 5 minutes
        'portfolio_hedge_analysis': 300,
        
        # Sparklines and charts - 15 minutes (900 seconds)
        'sparkline': 900,
        'sparklines': 900,
        'candlestick_chart': 300,
        'candlestick': 300,
        
        # Margin - 5 minutes
        'margin_data': 300,
        'margin/available': 300,
        'margin/calculate': 300,
    }
    
    # Find matching cache duration
    for key, duration in cache_durations.items():
        if key in endpoint_path:
            # Calculate expiration time
            expires_time = datetime.now() + timedelta(seconds=duration)
            expires_str = expires_time.strftime('%a, %d %b %Y %H:%M:%S GMT')
            return {
                'Cache-Control': f'public, max-age={duration}, s-maxage={duration}, stale-while-revalidate=60',
                'Expires': expires_str
            }
    
    # Default: 5 minutes for most endpoints
    default_duration = 300
    expires_time = datetime.now() + timedelta(seconds=default_duration)
    expires_str = expires_time.strftime('%a, %d %b %Y %H:%M:%S GMT')
    return {
        'Cache-Control': 'public, max-age=300, s-maxage=300, stale-while-revalidate=60',
        'Expires': expires_str
    }

def cached_json_response(data: dict, endpoint_path: str) -> JSONResponse:
    """
    Returns a JSONResponse with appropriate cache headers
    """
    headers = get_cache_headers(endpoint_path)
    return JSONResponse(content=data, headers=headers)
import io
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError as e:
    logging.error(f"TA-Lib not available: {e}")
    TALIB_AVAILABLE = False
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Function to calculate Supertrend
def calculate_supertrend(high, low, close, period=14, multiplier=3.0):
    """Calculate Supertrend indicator"""
    try:
        # Calculate ATR
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        # Calculate basic bands
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Initialize arrays
        supertrend = np.full(len(close), np.nan)
        direction = np.zeros(len(close))
        
        # Initialize final upper/lower bands
        final_upper_band = upper_band.copy()
        final_lower_band = lower_band.copy()
        
        # Calculate Supertrend
        for i in range(1, len(close)):
            # Final Upper Band
            if close[i-1] <= final_upper_band[i-1]:
                final_upper_band[i] = min(upper_band[i], final_upper_band[i-1])
            else:
                final_upper_band[i] = upper_band[i]
            
            # Final Lower Band
            if close[i-1] >= final_lower_band[i-1]:
                final_lower_band[i] = max(lower_band[i], final_lower_band[i-1])
            else:
                final_lower_band[i] = lower_band[i]
            
            # Supertrend
            if i == 1:
                # Initialize first value
                supertrend[i] = final_upper_band[i] if close[i-1] <= final_upper_band[i] else final_lower_band[i]
                direction[i] = -1 if close[i-1] <= final_upper_band[i] else 1
            else:
                if close[i-1] <= supertrend[i-1]:
                    supertrend[i] = final_upper_band[i]
                    direction[i] = -1  # Down
                else:
                    supertrend[i] = final_lower_band[i]
                    direction[i] = 1  # Up
        
        return supertrend, direction
    except Exception as e:
        logging.error(f"Error calculating Supertrend: {e}")
        return np.full(len(close), np.nan), np.zeros(len(close))

# Function to validate access token
def is_access_token_valid(access_token):
    print('cheking token')
    global valid_access_token
    try:
        kite.set_access_token(access_token) # Set the access token
        kite.margins()  # Make a simple API call to validate (e.g., get margins)
        
        logging.info("Existing access token is valid")
        valid_access_token = False
        return True
    except TokenException as e:
        valid_access_token = False
        logging.error(f"Access token validation failed: {e}")
        return False


# Function to generate a new access token
def generate_new_access_token(request_token):
    print('generate token')
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        redis_client.set("kite_access_token", access_token)
        redis_client.set("kite_access_token_timestamp", str(time.time()))
        logging.info(" --- XXXX ---")
        logging.info(f"New access token generated: {access_token}")
        logging.info(" --- XXXX ---")

        return access_token
    except Exception as e:
        logging.error(f"Failed to generate new access token: {e}")
        return None


# FastAPI app for capturing request_token
app = FastAPI()

# Template configuration
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Response compression to reduce payload size
app.add_middleware(GZipMiddleware, minimum_size=500)

# Global ANALYSIS_DATE configuration - can be set via API or environment
ANALYSIS_DATE = None  # Set to None for current date, or specify date like '2025-10-20'

# Global SHOW_HOLDINGS configuration - controls visibility of holdings sections
SHOW_HOLDINGS = os.getenv("SHOW_HOLDINGS", "True").lower() == "true"

# Small JSON cache helpers using Redis
def cache_get_json(key: str):
    try:
        val = redis_client.get(key)
        if not val:
            return None
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        return json.loads(val)
    except Exception:
        return None

def cache_set_json(key: str, value: dict, ttl_seconds: int = 5):
    try:
        redis_client.setex(key, ttl_seconds, json.dumps(value))
    except Exception:
        pass

# Function to check if access token was fetched today
def is_token_fetched_today():
    try:
        timestamp = redis_client.get("kite_access_token_timestamp")
        if not timestamp:
            return False
        
        token_time = datetime.fromtimestamp(float(timestamp))
        today = datetime.now().date()
        return token_time.date() == today
    except:
        return False

# Function to check if access token is currently valid
def is_token_currently_valid():
    try:
        access_token = redis_client.get("kite_access_token")
        if not access_token:
            return False
        return is_access_token_valid(access_token)
    except:
        return False

# Function to check tick data status
def get_tick_data_status():
    try:
        # Check if ticks are arriving by looking at recent data
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check for ticks in the last 5 minutes
        cursor.execute("""
            SELECT COUNT(*) FROM my_schema.ticks 
            WHERE timestamp > NOW() - INTERVAL '5 minutes'
            AND instrument_token = 256265
        """)
        recent_ticks = cursor.fetchone()[0]
        
        # Get total ticks for today
        cursor.execute("""
            SELECT COUNT(*) FROM my_schema.ticks 
            WHERE DATE(timestamp + INTERVAL '5 hours 30 minutes') = CURRENT_DATE
            AND instrument_token = 256265
        """)
        total_ticks = cursor.fetchone()[0]
        
        # Get latest tick timestamp as pre-formatted IST string directly from DB to avoid timezone drift
        cursor.execute("""
            SELECT 
                to_char(MAX(timestamp AT TIME ZONE 'Asia/Kolkata'), 'YYYY-MM-DD HH24:MI:SS') AS latest_ts_ist_str
            FROM my_schema.ticks
            WHERE instrument_token = 256265
        """)
        latest_row = cursor.fetchone()
        latest_ts_ist_str = latest_row[0] if latest_row else None

        # Debug: log DB output
        logging.info(f"TickStatus DB latest_ts_ist_str={latest_ts_ist_str}, recent_ticks={recent_ticks}, total_ticks={total_ticks}")

        conn.close()
        
        return {
            'active': recent_ticks > 0,
            'recent_ticks': recent_ticks,
            'total_ticks': total_ticks,
            'latest_tick_str': latest_ts_ist_str
        }
    except Exception as e:
        logging.error(f"Error checking tick data status: {e}")
        return {
            'active': False,
            'recent_ticks': 0,
            'total_ticks': 0,
            'latest_tick_str': None
        }


# Function to get holdings data
def get_holdings_data(page: int = 1, per_page: int = 10, sort_by: str = None, sort_dir: str = "asc", search: str = None):
    """Fetch current holdings from the database with pagination and sorting."""
    try:
        conn = get_db_connection()
        # Use RealDictCursor to get results as dictionaries
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Build WHERE clause with search filter
        where_clause = "WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)"
        search_params = []
        if search and search.strip():
            where_clause += " AND h.trading_symbol ILIKE %s"
            search_params.append(f"%{search.strip()}%")

        # Fetch holdings for the most recent run_date
        # Get total count for pagination
        count_query = f"""
            SELECT COUNT(*) as total_count
            FROM my_schema.holdings h
            {where_clause}
        """
        cursor.execute(count_query, search_params)
        total_count = cursor.fetchone()['total_count']

        # Validate and set sort column
        valid_sort_columns = ['trading_symbol', 'invested_amount', 'current_amount', 'pnl', 'today_pnl', 'prophet_prediction_pct']
        if sort_by not in valid_sort_columns:
            sort_by = 'trading_symbol'
        
        # Validate sort direction
        if sort_dir.lower() not in ['asc', 'desc']:
            sort_dir = 'asc'

        # Calculate offset for pagination
        offset = (page - 1) * per_page

        # Fetch paginated holdings for the most recent run_date
        # Calculate invested amount and current value with coalesced prices
        # Include Prophet predictions from latest run_date
        cursor.execute("""
            SELECT 
                h.trading_symbol,
                h.instrument_token,
                h.quantity,
                h.average_price,
                h.last_price,
                COALESCE(h.last_price, rt.price_close) as current_price,
                h.pnl,
                (h.quantity * h.average_price) as invested_amount,
                (h.quantity * COALESCE(h.last_price, rt.price_close)) as current_amount,
                round(case when (h.quantity * h.average_price) != 0 then
                    (h.pnl / (h.quantity * h.average_price)) * 100
                else 0
                end::numeric, 2) as pnl_pct_change,
                pp.predicted_price_change_pct,
                pp.prediction_confidence,
                (SELECT MAX(run_date) FROM my_schema.holdings) as run_date
            FROM my_schema.holdings h
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE price_date::date <= CURRENT_DATE
                ORDER BY scrip_id, price_date DESC
            ) rt ON h.trading_symbol = rt.scrip_id
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, predicted_price_change_pct, prediction_confidence
                FROM my_schema.prophet_predictions
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE')
                AND prediction_days = 30
                AND status = 'ACTIVE'
                ORDER BY scrip_id, run_date DESC
            ) pp ON h.trading_symbol = pp.scrip_id
            {where_clause}
            ORDER BY {sort_by} {sort_dir}
            LIMIT %s OFFSET %s
        """.format(where_clause=where_clause, sort_by=sort_by, sort_dir=sort_dir.upper()), search_params + [per_page, offset])
        holdings = cursor.fetchall()
        
        # Convert RealDictRow objects to regular dictionaries for Jinja2 template compatibility
        holdings_list = [dict(row) for row in holdings]
        
        conn.close()
        return {
            "holdings": holdings_list,
            "total_count": total_count,
            "page": page,
            "per_page": per_page
        }
    except Exception as e:
        logging.error(f"Error fetching holdings data: {e}")
        return {
            "holdings": [],
            "total_count": 0,
            "page": page,
            "per_page": per_page
        }


# Function to get system status
def get_system_status():
    # Get current IST time
    from datetime import datetime, timedelta
    ist_now = datetime.now() + timedelta(hours=5, minutes=30)
    
    tick_status = get_tick_data_status()
    last_update_value = tick_status.get('latest_tick_str') or ist_now.strftime('%H:%M:%S IST')
    # Debug: log what will be sent to UI
    logging.info(
        f"SystemStatus last_update={last_update_value}, tick_active={tick_status['active']}, "
        f"total_ticks={tick_status['total_ticks']}, raw_latest={tick_status.get('latest_tick_str')}"
    )
    
    return {
        'token_fetched_today': is_token_fetched_today(),
        'token_valid': is_token_currently_valid(),
        'tick_data': tick_status,
        'last_update': last_update_value
    }

# Helper function to enrich holdings with today's P&L
def enrich_holdings_with_today_pnl(holdings_data):
    """Enrich holdings data with today's P&L from rt_intraday_price"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get latest available date (not necessarily today - could be last trading day)
        cursor.execute("""
            SELECT MAX(price_date::date) as latest_date
            FROM my_schema.rt_intraday_price
        """)
        latest_result = cursor.fetchone()
        latest_date = latest_result['latest_date'] if latest_result and latest_result['latest_date'] else None
        
        if not latest_date:
            logging.warning("No data found in rt_intraday_price, returning holdings without today's P&L")
            conn.close()
            # Return holdings with default today_pnl values
            enriched_holdings = []
            for holding in holdings_data.get('holdings', []):
                holding_dict = dict(holding)
                holding_dict['today_pnl'] = 0.0
                holding_dict['today_price'] = 0.0
                holding_dict['prev_price'] = 0.0
                holding_dict['pct_change'] = 0.0
                holding_dict['today_change'] = '0 (0%)'
                enriched_holdings.append(holding_dict)
            holdings_data['holdings'] = enriched_holdings
            return holdings_data
        
        # Convert latest_date to string
        latest_date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
        
        # Get previous trading day
        cursor.execute("""
            SELECT MAX(price_date::date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date::date < %s
        """, (latest_date,))
        
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        logging.info(f"Using latest date: {latest_date_str}, previous date: {prev_date}")
        
        # Get today's P&L for all holdings - match by instrument_token
        today_pnl_map = {}
        
        # Always populate the map with holdings, even if prev_date doesn't exist (will default to 0)
        # First, get all holdings to ensure we have entries for all of them
        cursor.execute("""
            SELECT instrument_token, trading_symbol, quantity
            FROM my_schema.holdings
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
        """)
        
        for row in cursor.fetchall():
            instrument_token = row['instrument_token']
            # Initialize with 0 values for all holdings
            today_pnl_map[instrument_token] = {
                'today_pnl': 0.0,
                'today_price': 0.0,
                'prev_price': 0.0,
                'pct_change': 0.0,
                'today_change': '0 (0%)'
            }
        
        if prev_date:
            # Convert prev_date to string format
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT instrument_token, trading_symbol, quantity, last_price
                    FROM my_schema.holdings
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close, price_date
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                ),
                latest_prices AS (
                    SELECT scrip_id, price_close, 
                           ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= %s
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                )
                SELECT 
                    h.instrument_token,
                    h.trading_symbol,
                    h.quantity,
                    COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    -- If today's price is null/0, P&L should be 0 regardless of prev price
                    CASE 
                        WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                        ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                    END as today_pnl,
                    round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
	                    ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
	                else 0
	                end::numeric, 2) as pct_change,
                    concat(round(CASE 
                        WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                        ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                    END), ' (',
                    round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
	                    ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
	                else 0
	                end::numeric, 2), '%%)') as "Todays_Change"
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
                
            """, (latest_date, latest_date, prev_date))
            
            for row in cursor.fetchall():
                instrument_token = row['instrument_token']
                today_price = float(row['today_price']) if row['today_price'] else 0.0
                prev_price = float(row['prev_price']) if row['prev_price'] else 0.0
                today_pnl = float(row['today_pnl']) if row['today_pnl'] else 0.0
                pct_change = float(row['pct_change']) if row['pct_change'] else 0.0
                today_change_str = str(row.get("Todays_Change", '')) if row.get("Todays_Change") else ''
                
                # Store by instrument_token to match with holdings
                today_pnl_map[instrument_token] = {
                    'today_pnl': today_pnl,
                    'today_price': today_price,
                    'prev_price': prev_price,
                    'pct_change': pct_change,
                    'today_change': today_change_str
                }
        
        # Get Prophet predictions for all holdings (60-day predictions)
        try:
            # First try to get 60-day predictions
            cursor.execute("""
                SELECT scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
                FROM my_schema.prophet_predictions
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE' AND prediction_days = 60)
                AND prediction_days = 60
                AND status = 'ACTIVE'
            """)
            
            predictions_rows = cursor.fetchall()
            
            # If no 60-day predictions, try to get latest predictions regardless of prediction_days
            if not predictions_rows:
                logging.warning("No 60-day Prophet predictions found for holdings, trying latest predictions")
                cursor.execute("""
                    SELECT scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
                    FROM my_schema.prophet_predictions pp1
                    WHERE status = 'ACTIVE'
                    AND run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE')
                """)
                predictions_rows = cursor.fetchall()
            
            predictions_map = {row['scrip_id'].upper(): dict(row) for row in predictions_rows}
            logging.info(f"Loaded {len(predictions_map)} Prophet predictions for holdings enrichment")
            
            if predictions_map:
                sample_ids = list(predictions_map.keys())[:5]
                logging.debug(f"Sample Prophet prediction scrip_ids for holdings: {sample_ids}")
        except Exception as e:
            logging.error(f"Error loading Prophet predictions for holdings: {e}")
            import traceback
            logging.error(traceback.format_exc())
            predictions_map = {}
        
        conn.close()
        
        # Convert RealDictRow objects to regular dictionaries and enrich with today's P&L and Prophet predictions
        enriched_holdings = []
        matched_predictions = 0
        holdings_symbols = []
        
        for holding in holdings_data.get('holdings', []):
            # Convert RealDictRow to dict
            holding_dict = dict(holding)
            instrument_token = holding_dict.get('instrument_token')
            symbol = holding_dict.get('trading_symbol', '')
            holdings_symbols.append(symbol.upper() if symbol else '')
            
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'today_price': 0.0, 'prev_price': 0.0, 'pct_change': 0.0, 'today_change': '0 (0%)'})
            
            # Add today's P&L fields
            holding_dict['today_pnl'] = today_pnl_info['today_pnl']
            holding_dict['today_price'] = today_pnl_info['today_price']
            holding_dict['prev_price'] = today_pnl_info['prev_price']
            holding_dict['pct_change'] = today_pnl_info.get('pct_change', 0.0)
            holding_dict['today_change'] = today_pnl_info.get('today_change', '0 (0%)')
            
            # Get Prophet prediction for this holding
            prophet_pred = None
            prophet_conf = None
            prediction_days = None
            if symbol:
                symbol_upper = symbol.upper()
                if symbol_upper in predictions_map:
                    pred = predictions_map[symbol_upper]
                    try:
                        pred_pct = pred.get('predicted_price_change_pct')
                        pred_conf = pred.get('prediction_confidence')
                        pred_days = pred.get('prediction_days')
                        
                        if pred_pct is not None and not (isinstance(pred_pct, float) and (pred_pct != pred_pct)):  # Check for NaN
                            prophet_pred = float(pred_pct)
                        if pred_conf is not None and not (isinstance(pred_conf, float) and (pred_conf != pred_conf)):  # Check for NaN
                            prophet_conf = float(pred_conf)
                        if pred_days is not None:
                            prediction_days = int(pred_days)
                        
                        if prophet_pred is not None:
                            matched_predictions += 1
                    except (ValueError, TypeError) as e:
                        logging.debug(f"Error converting Prophet prediction for {symbol}: {e}")
                        prophet_pred = None
                        prophet_conf = None
                        prediction_days = None
                else:
                    logging.debug(f"No Prophet prediction found for holding {symbol} (searched as: {symbol_upper})")
            
            holding_dict['prophet_prediction_pct'] = prophet_pred
            holding_dict['prophet_confidence'] = prophet_conf
            holding_dict['prediction_days'] = prediction_days
            
            # Ensure pnl_pct_change field exists (from original query)
            if 'pnl_pct_change' not in holding_dict:
                # Calculate it if missing
                invested_amount = holding_dict.get('invested_amount', 0)
                pnl = holding_dict.get('pnl', 0)
                holding_dict['pnl_pct_change'] = (pnl / invested_amount * 100) if invested_amount != 0 else 0.0
            
            enriched_holdings.append(holding_dict)
        
        # Log matching statistics
        if len(enriched_holdings) > 0:
            if matched_predictions > 0:
                logging.info(f"✓ Matched {matched_predictions}/{len(enriched_holdings)} holdings with Prophet predictions")
            else:
                logging.warning(f"⚠ No Prophet predictions matched for holdings!")
                logging.warning(f"  - Holdings symbols: {sorted(set(holdings_symbols))[:10]}{'...' if len(set(holdings_symbols)) > 10 else ''}")
                logging.warning(f"  - Prophet prediction scrip_ids: {sorted(list(predictions_map.keys()))[:10] if predictions_map else []}{'...' if len(predictions_map) > 10 else ''}")
        
        # Update holdings_data with enriched holdings
        holdings_data['holdings'] = enriched_holdings
        return holdings_data
        
    except Exception as e:
        logging.error(f"Error enriching holdings with today's P&L: {e}")
        # Return holdings with default today_pnl values and Prophet predictions
        enriched_holdings = []
        for holding in holdings_data.get('holdings', []):
            holding_dict = dict(holding)
            holding_dict['today_pnl'] = 0.0
            holding_dict['today_price'] = 0.0
            holding_dict['prev_price'] = 0.0
            holding_dict['pct_change'] = 0.0
            holding_dict['today_change'] = '0 (0%)'
            
            # Default Prophet predictions to None
            holding_dict['prophet_prediction_pct'] = None
            holding_dict['prophet_confidence'] = None
            holding_dict['prediction_days'] = None
            
            # Ensure pnl_pct_change field exists
            if 'pnl_pct_change' not in holding_dict:
                invested_amount = holding_dict.get('invested_amount', 0)
                pnl = holding_dict.get('pnl', 0)
                holding_dict['pnl_pct_change'] = (pnl / invested_amount * 100) if invested_amount != 0 else 0.0
            
            enriched_holdings.append(holding_dict)
        
        holdings_data['holdings'] = enriched_holdings
        return holdings_data

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, page: int = Query(1, ge=1)):
    """Dashboard route - shows dashboard if valid token exists"""
    # Check if valid access token exists in Redis
    existing_token = redis_client.get("kite_access_token")
    if existing_token:
        # Handle both string and bytes from Redis
        if isinstance(existing_token, bytes):
            existing_token = existing_token.decode('utf-8')
        
        # Check if token is still valid
        if is_access_token_valid(existing_token):
            logging.info("Valid access token found, showing dashboard")
            
            # Get system status for dashboard
            system_status = get_system_status()
            tick_data = system_status['tick_data']
            # Debug: log values being passed to template
            logging.info(
                f"DashboardRender last_update={system_status['last_update']}, total_ticks={tick_data['total_ticks']}"
            )
            holdings_info = get_holdings_data(page=page, per_page=10)
            
            # Enrich holdings with today's P&L
            holdings_info = enrich_holdings_with_today_pnl(holdings_info)
            
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "token_status": "Valid" if system_status['token_valid'] else "Invalid",
                "tick_status": "Active" if tick_data['active'] else "Inactive",
                "last_update": system_status['last_update'],
                "total_ticks": tick_data['total_ticks'],
                "holdings_info": holdings_info,
                "show_holdings": SHOW_HOLDINGS
            })
    
    # No valid token, redirect to login
    login_url = kite.login_url()
    return templates.TemplateResponse("login.html", {
        "request": request,
        "login_url": login_url
    })

@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    action: str = Query(None),
    type: str = Query(None),
    status: str = Query(None),
    request_token: str = Query(None)
):
    global token_access
    
    if request_token and request_token.strip() and status == "success" and action == "login" and type == "login":
        # Handle redirect from Zerodha with request_token
        redis_client.set("kite_request_token", request_token)
        logging.info(f"Received request_token: {request_token}")
        token_access = request_token

        # Trigger get_access_token to generate and save access token
        # This will reuse existing token if valid, or generate new one
        try:
            access_token = get_access_token()  # Call your function here
        except Exception as e:
            logging.error(f"Error getting access token: {e}")
            # If getting token fails, show login page
            login_url = kite.login_url()
            return templates.TemplateResponse("login.html", {
                "request": request,
                "login_url": login_url
            })

        # Get system status for dashboard
        system_status = get_system_status()
        tick_data = system_status['tick_data']
        holdings_info = get_holdings_data(page=1, per_page=10)
        
        # Enrich holdings with today's P&L and Prophet predictions
        holdings_info = enrich_holdings_with_today_pnl(holdings_info)
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "token_status": "Valid" if system_status['token_valid'] else "Invalid",
            "tick_status": "Active" if tick_data['active'] else "Inactive",
            "last_update": system_status['last_update'],
            "total_ticks": tick_data['total_ticks'],
            "holdings_info": holdings_info,
            "show_holdings": SHOW_HOLDINGS
        })
    else:
        # Check if there's an existing valid token
        existing_token = redis_client.get("kite_access_token")
        if existing_token:
            # Handle both string and bytes from Redis
            if isinstance(existing_token, bytes):
                existing_token = existing_token.decode('utf-8')
            
            # Check if token is still valid
            if is_access_token_valid(existing_token):
                logging.info("Valid access token found, redirecting to dashboard")
                # Redirect to dashboard instead of showing login
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url="/dashboard", status_code=302)
        
        # Show login page
        login_url = kite.login_url()
        return templates.TemplateResponse("login.html", {
            "request": request,
            "login_url": login_url
        })


# API endpoints for dashboard
@app.get("/api/system_status")
async def api_system_status():
    """API endpoint to get current system status"""
    status = get_system_status()
    return cached_json_response({
        "token_valid": status['token_valid'],
        "token_fetched_today": status['token_fetched_today'],
        "tick_active": status['tick_data']['active'],
        "recent_ticks": status['tick_data']['recent_ticks'],
        "total_ticks": status['tick_data']['total_ticks'],
        "last_update": status['last_update']
    }, "/api/system_status")

@app.get("/api/tpo_charts")
async def api_tpo_charts(analysis_date: str = Query(None)):
    """API endpoint to generate TPO chart images"""
    try:
        from market.CalculateTPO import PostgresDataFetcher, TPOProfile
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import base64
        
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
        
        # Get real-time TPO data
        if analysis_date or ANALYSIS_DATE:
            end_time = f'{target_date} 15:30:00.000 +0530'
        else:
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.000 +0530")
        
        real_time_df = db_fetcher.fetch_tick_data(
            table_name='ticks',
            instrument_token=256265,
            start_time=f'{target_date} 09:15:00.000 +0530',
            end_time=f'{target_date} 15:30:00.000 +0530'
        )
        
        # Generate charts with dark background
        fig = plt.figure(figsize=(24, 10), facecolor='black')
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        # Pre-market chart
        if not pre_market_df.empty:
            pre_market_tpo = TPOProfile(tick_size=5)
            pre_market_tpo.calculate_tpo(pre_market_df)
            pre_market_tpo.plot_profile(ax=ax1, show_metrics=True, show_letters=True, dark_mode=True)
            ax1.set_title(f"Pre-market TPO Profile ({target_date})", fontsize=14, fontweight='bold', color='white')
        else:
            ax1.set_facecolor('black')
            ax1.text(0.5, 0.5, 'No pre-market data', ha='center', va='center', transform=ax1.transAxes, color='white', fontsize=12)
            ax1.set_title(f"Pre-market TPO Profile ({target_date})", fontsize=14, fontweight='bold', color='white')
            ax1.tick_params(colors='white', labelsize=10)
            ax1.spines['top'].set_color('white')
            ax1.spines['bottom'].set_color('white')
            ax1.spines['left'].set_color('white')
            ax1.spines['right'].set_color('white')
            ax1.xaxis.label.set_color('white')
            ax1.yaxis.label.set_color('white')
        
        # Real-time chart
        if not real_time_df.empty:
            real_time_tpo = TPOProfile(tick_size=5)
            real_time_tpo.calculate_tpo(real_time_df)
            real_time_tpo.plot_profile(ax=ax2, show_metrics=True, show_letters=True, dark_mode=True)
            ax2.set_title(f"Real-time TPO Profile ({target_date})", fontsize=14, fontweight='bold', color='white')
        else:
            ax2.set_facecolor('black')
            ax2.text(0.5, 0.5, 'No real-time data', ha='center', va='center', transform=ax2.transAxes, color='white', fontsize=12)
            ax2.set_title(f"Real-time TPO Profile ({target_date})", fontsize=14, fontweight='bold', color='white')
            ax2.tick_params(colors='white', labelsize=10)
            ax2.spines['top'].set_color('white')
            ax2.spines['bottom'].set_color('white')
            ax2.spines['left'].set_color('white')
            ax2.spines['right'].set_color('white')
            ax2.xaxis.label.set_color('white')
            ax2.yaxis.label.set_color('white')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='black')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "analysis_date": target_date,
            "is_historical": bool(analysis_date or ANALYSIS_DATE),
            "chart_image": f"data:image/png;base64,{image_base64}"
        }
    except Exception as e:
        logging.error(f"Error generating TPO charts: {e}")
        return {"error": str(e)}

@app.get("/api/tpo_5day_chart")
async def api_tpo_5day_chart(analysis_date: str = Query(None)):
    """API endpoint to generate 5-day TPO chart with volume profiles"""
    try:
        from market.CalculateTPO import PostgresDataFetcher, plot_5day_tpo_chart
        import matplotlib
        matplotlib.use('Agg')
        
        DB_CONFIG = {
            'host': 'postgres',
            'database': 'mydb',
            'user': 'postgres',
            'password': 'postgres',
            'port': 5432
        }
        
        db_fetcher = PostgresDataFetcher(**DB_CONFIG)
        
        # Generate 5-day TPO chart
        chart_image = plot_5day_tpo_chart(
            db_fetcher=db_fetcher,
            table_name='ticks',
            instrument_token=256265,
            tick_size=5,
            market_start_time="09:15",
            market_end_time="15:30"
        )
        
        if chart_image:
            return {
                "success": True,
                "chart_image": chart_image,
                "message": "5-day TPO chart generated successfully"
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate 5-day TPO chart - no data available"
            }
        
    except Exception as e:
        logging.error(f"Error generating 5-day TPO chart: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/analysis_date")
async def get_analysis_date():
    """Get current ANALYSIS_DATE configuration"""
    return {
        "analysis_date": ANALYSIS_DATE,
        "is_live_mode": ANALYSIS_DATE is None,
        "current_date": datetime.now().strftime("%Y-%m-%d")
    }

@app.post("/api/analysis_date")
async def set_analysis_date(date: str = Query(None)):
    """Set ANALYSIS_DATE for backtesting"""
    global ANALYSIS_DATE
    
    if date is None or date == "live" or date == "":
        ANALYSIS_DATE = None
        return {
            "message": "Switched to live mode",
            "analysis_date": None,
            "is_live_mode": True
        }
    
    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
        ANALYSIS_DATE = date
        return {
            "message": f"Analysis date set to {date}",
            "analysis_date": date,
            "is_live_mode": False
        }
    except ValueError:
        return {
            "error": "Invalid date format. Use YYYY-MM-DD format",
            "analysis_date": ANALYSIS_DATE,
            "is_live_mode": ANALYSIS_DATE is None
        }

@app.get("/api/available_dates")
async def get_available_dates():
    """Get list of available dates in the database for backtesting"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get available dates from ticks table
        cursor.execute("""
            SELECT DISTINCT DATE(timestamp + INTERVAL '5 hours 30 minutes') as trade_date
            FROM my_schema.ticks 
            WHERE instrument_token = 256265
            ORDER BY trade_date DESC
            LIMIT 30
        """)
        
        dates = [row[0].strftime("%Y-%m-%d") for row in cursor.fetchall()]
        conn.close()
        
        return {
            "available_dates": dates,
            "current_analysis_date": ANALYSIS_DATE
        }
    except Exception as e:
        logging.error(f"Error fetching available dates: {e}")
        return {"error": str(e)}

@app.post("/api/refresh_futures")
async def api_refresh_futures():
    """API endpoint to refresh futures data"""
    try:
        from kite.KiteFetchFuture import fetch_and_save_futures
        result = fetch_and_save_futures()
        return result
    except Exception as e:
        logging.error(f"Error refreshing futures data: {e}")
        return {"success": False, "error": str(e)}

@app.get("/refresh_data")
async def refresh_data():
    """Endpoint to refresh all data including futures"""
    try:
        from kite.KiteFetchFuture import fetch_and_save_futures
        futures_result = fetch_and_save_futures()
        return {
            "message": "Data refresh initiated", 
            "status": "success",
            "futures": futures_result
        }
    except Exception as e:
        logging.error(f"Error in refresh_data: {e}")
        return {"message": "Data refresh initiated", "status": "partial", "error": str(e)}

@app.get("/logout")
async def logout():
    """Endpoint to logout and clear tokens"""
    redis_client.delete("kite_access_token")
    redis_client.delete("kite_access_token_timestamp")
    redis_client.delete("kite_request_token")
    return {"message": "Logged out successfully", "status": "success"}

@app.get("/redirect", response_class=HTMLResponse)
async def handle_redirect(request_token: str = None):
    logging.info('redirects')

    if not request_token:
        raise HTTPException(status_code=400, detail="No request_token provided")

    # Store request_token in Redis
    redis_client.set("kite_request_token", request_token)
    logging.info(f"Received request_token: {request_token}")

    # Trigger token generation
    #access_token = get_access_token()

    return f"""
        <h1>Kite Connect Authentication</h1>
        <p>Request token received: {request_token}</p>
        <p>Access token generation in progress...</p>
    """


# Main logic
# Note here that if a valid access token is available in Redis then login is not necessary and the all the code would run successfully
#
def get_access_token():
    logging.info('get access token')
    global ACCESS_TOKEN
    
    # First, check if we already have a valid access token
    existing_token = redis_client.get("kite_access_token")
    if existing_token:
        # Handle both string and bytes from Redis
        if isinstance(existing_token, bytes):
            existing_token = existing_token.decode('utf-8')
        
        try:
            # Validate the existing token
            if is_access_token_valid(existing_token):
                logging.info("Using existing valid access token")
                ACCESS_TOKEN = existing_token
                return existing_token
        except TokenException:
            logging.warning("Existing token is invalid, will generate new one")
    
    ACCESS_TOKEN = ''
    
    # Check if there's already a request_token in Redis
    request_token = redis_client.get("kite_request_token")
    
    # If no request_token exists, wait for it to be set
    if not request_token or not request_token.strip():
    # Clear any existing request_token in Redis
        redis_client.delete("kite_request_token")
    redis_client.delete("kite_access_token")
    redis_client.delete("kite_access_token_timestamp")
    logging.info('Waiting for request_token...')
    
    # Wait for request_token from Redis
    timeout = 600  # 10 minutes
    start_time = time.time()
    while time.time() - start_time < timeout:
        request_token = redis_client.get("kite_request_token")
        if request_token and request_token.strip():
            break
        time.sleep(1)  # Poll every second
    else:
        raise Exception("Failed to receive request_token within timeout")
    
    # Generate new access token
    logging.info("Calling generate new access token")
    new_access_token = generate_new_access_token(request_token)
    if new_access_token:
        ACCESS_TOKEN = new_access_token
        return new_access_token
    else:
        raise Exception("Could not obtain a valid access token")


@app.get("/api/market_dashboard")
async def api_market_dashboard(analysis_date: str = Query(None)):
    """API endpoint to generate complete market dashboard"""
    try:
        return {
            "analysis_date": analysis_date or datetime.now().strftime("%Y-%m-%d"),
            "is_historical": bool(analysis_date or ANALYSIS_DATE),
            "dashboard_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        }
    except Exception as e:
        logging.error(f"Error generating market dashboard: {e}")
        return {"error": str(e)}

@app.get("/api/candlestick_chart")
async def api_candlestick_chart(analysis_date: str = Query(None), chart_type: str = Query("market")):
    """API endpoint to generate candlestick chart"""
    try:
        return {
            "analysis_date": analysis_date or datetime.now().strftime("%Y-%m-%d"),
            "chart_type": chart_type,
            "is_historical": bool(analysis_date or ANALYSIS_DATE),
            "chart_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        }
    except Exception as e:
        logging.error(f"Error generating candlestick chart: {e}")
        return {"error": str(e)}

@app.get("/api/trades_table")
async def api_trades_table():
    """API endpoint to generate trades table"""
    try:
        return {
            "trades_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        }
    except Exception as e:
        logging.error(f"Error generating trades table: {e}")
        return {"error": str(e)}

@app.get("/api/gainers_losers")
async def api_gainers_losers():
    """API endpoint to generate gainers and losers chart"""
    try:
        return {
            "gainers_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        }
    except Exception as e:
        logging.error(f"Error generating gainers/losers chart: {e}")
        return {"error": str(e)}

@app.get("/api/margin_data")
async def api_margin_data():
    """API endpoint to get margin data for status bar"""
    try:
        # Use the existing database connection from Boilerplate
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get margin data from database
        cursor.execute("""
            SELECT DISTINCT margin_type, net, available_cash, available_live_balance
            FROM my_schema.margins
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.margins)
            AND enabled IS TRUE
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            total_cash = sum(row[2] for row in results if row[2] is not None)
            total_live_balance = sum(row[3] for row in results if row[3] is not None)
        else:
            total_cash = 0
            total_live_balance = 0
        
        return {
            "available_cash": total_cash,
            "live_balance": total_live_balance
        }
    except Exception as e:
        logging.error(f"Error fetching margin data: {e}")
        return {
            "available_cash": 0,
            "live_balance": 0
        }

@app.get("/api/market_bias")
async def api_market_bias(analysis_date: str = Query(None)):
    """API endpoint to get market bias analysis"""
    try:
        from market.MarketBiasAnalyzer import MarketBiasAnalyzer, PostgresDataFetcher
        
        DB_CONFIG = {
            'host': 'postgres',
            'database': 'mydb',
            'user': 'postgres',
            'password': 'postgres',
            'port': 5432
        }
        
        db_fetcher = PostgresDataFetcher(**DB_CONFIG)
        bias_analyzer = MarketBiasAnalyzer(db_fetcher, instrument_token=256265, tick_size=5.0)
        
        # Determine analysis date
        if analysis_date:
            target_date = analysis_date
        elif ANALYSIS_DATE:
            target_date = ANALYSIS_DATE
        else:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        # Generate comprehensive analysis
        analysis = bias_analyzer.generate_comprehensive_analysis(target_date)
        
        return {
            "analysis_date": target_date,
            "is_historical": bool(analysis_date or ANALYSIS_DATE),
            "analysis": analysis
        }
    except Exception as e:
        logging.error(f"Error generating market bias analysis: {e}")
        return {"error": str(e)}

@app.get("/api/market_bias_chart")
async def api_market_bias_chart(analysis_date: str = Query(None)):
    """API endpoint to get market bias analysis chart"""
    try:
        from market.MarketBiasAnalyzer import MarketBiasAnalyzer, PostgresDataFetcher
        
        DB_CONFIG = {
            'host': 'postgres',
            'database': 'mydb',
            'user': 'postgres',
            'password': 'postgres',
            'port': 5432
        }
        
        db_fetcher = PostgresDataFetcher(**DB_CONFIG)
        bias_analyzer = MarketBiasAnalyzer(db_fetcher, instrument_token=256265, tick_size=5.0)
        
        # Determine analysis date
        if analysis_date:
            target_date = analysis_date
        elif ANALYSIS_DATE:
            target_date = ANALYSIS_DATE
        else:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        # Generate comprehensive analysis
        analysis = bias_analyzer.generate_comprehensive_analysis(target_date)
        
        # Generate plot
        plot_image = bias_analyzer.plot_bias_analysis(analysis)
        
        return {
            "analysis_date": target_date,
            "is_historical": bool(analysis_date or ANALYSIS_DATE),
            "chart_image": plot_image
        }
    except Exception as e:
        logging.error(f"Error generating market bias chart: {e}")
        return {"error": str(e)}

@app.get("/api/premarket_analysis")
async def api_premarket_analysis(analysis_date: str = Query(None)):
    """API endpoint to get comprehensive pre-market TPO analysis"""
    try:
        from market.PremarketAnalyzer import PremarketAnalyzer
        from market.CalculateTPO import PostgresDataFetcher
        
        DB_CONFIG = {
            'host': 'postgres',
            'database': 'mydb',
            'user': 'postgres',
            'password': 'postgres',
            'port': 5432
        }
        
        db_fetcher = PostgresDataFetcher(**DB_CONFIG)
        premarket_analyzer = PremarketAnalyzer(db_fetcher, instrument_token=256265, tick_size=5.0)
        
        # Determine analysis date
        if analysis_date:
            target_date = analysis_date
        elif ANALYSIS_DATE:
            target_date = ANALYSIS_DATE
        else:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        # Generate comprehensive pre-market analysis
        analysis = premarket_analyzer.generate_comprehensive_premarket_analysis(target_date)
        
        # Ensure success field exists
        if isinstance(analysis, dict) and 'success' not in analysis:
            analysis['success'] = True
        
        return analysis
    except Exception as e:
        logging.error(f"Error generating pre-market analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        # Return user-friendly error message
        error_msg = "Unable to generate pre-market analysis. Please try again later."
        if "DataFrame" in str(e) or "ambiguous" in str(e).lower():
            error_msg = "Data processing error in pre-market analysis. Please contact support if this persists."
        elif "connection" in str(e).lower() or "timeout" in str(e).lower():
            error_msg = "Connection error while fetching pre-market data. Please check your connection and try again."
        elif "data" in str(e).lower() and ("not found" in str(e).lower() or "empty" in str(e).lower()):
            error_msg = "Pre-market data is not available for the selected date. Please select a different date."
        
        return {
            "success": False, 
            "error": error_msg,
            "analysis_date": target_date if 'target_date' in locals() else None
        }

@app.get("/api/footprint_analysis")
async def api_footprint_analysis(
    start_time: str = Query("09:15:00", description="Start time in HH:MM:SS format"),
    end_time: str = Query("15:30:00", description="End time in HH:MM:SS format"),
    analysis_date: str = Query(None, description="Analysis date in YYYY-MM-DD format (default: today)"),
    instrument_token: int = Query(12683010, description="Instrument token (default: 12683010 for Nifty 50 Futures)")
):
    """API endpoint for footprint chart analysis"""
    try:
        from market.FootprintChartGenerator import FootprintChartGenerator
        from datetime import datetime, date
        
        footprint_gen = FootprintChartGenerator(instrument_token=instrument_token)
        
        # Parse analysis date
        if analysis_date:
            try:
                target_date = datetime.strptime(analysis_date, '%Y-%m-%d').date()
            except:
                target_date = date.today()
        elif ANALYSIS_DATE:
            target_date = datetime.strptime(ANALYSIS_DATE, '%Y-%m-%d').date()
        else:
            target_date = date.today()
        
        # Generate footprint data
        footprint_data = footprint_gen.generate_footprint_data(
            start_time=start_time,
            end_time=end_time,
            analysis_date=target_date
        )
        
        return footprint_data
    except Exception as e:
        logging.error(f"Error generating footprint analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.get("/api/orderflow_analysis")
async def api_orderflow_analysis(
    start_time: str = Query("09:15:00", description="Start time in HH:MM:SS format"),
    end_time: str = Query("15:30:00", description="End time in HH:MM:SS format"),
    analysis_date: str = Query(None, description="Analysis date in YYYY-MM-DD format (default: today)"),
    instrument_token: int = Query(12683010, description="Instrument token (default: 12683010 for Nifty 50 Futures)")
):
    """API endpoint for order flow analysis"""
    try:
        from market.OrderFlowAnalyzer import OrderFlowAnalyzer
        from datetime import datetime, date
        
        orderflow_analyzer = OrderFlowAnalyzer(instrument_token=instrument_token)
        
        # Parse analysis date
        if analysis_date:
            try:
                target_date = datetime.strptime(analysis_date, '%Y-%m-%d').date()
            except:
                target_date = date.today()
        elif ANALYSIS_DATE:
            target_date = datetime.strptime(ANALYSIS_DATE, '%Y-%m-%d').date()
        else:
            target_date = date.today()
        
        # Analyze order flow
        analysis = orderflow_analyzer.analyze_order_flow(
            start_time=start_time,
            end_time=end_time,
            analysis_date=target_date
        )
        
        return analysis
    except Exception as e:
        logging.error(f"Error analyzing order flow: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.get("/api/micro_levels")
async def api_micro_levels(
    start_time: str = Query("09:15:00", description="Start time in HH:MM:SS format"),
    end_time: str = Query("15:30:00", description="End time in HH:MM:SS format"),
    analysis_date: str = Query(None, description="Analysis date in YYYY-MM-DD format (default: today)"),
    instrument_token: int = Query(12683010, description="Instrument token (default: 12683010 for Nifty 50 Futures)")
):
    """API endpoint for micro level detection"""
    try:
        from market.MicroLevelDetector import MicroLevelDetector
        from datetime import datetime, date
        
        detector = MicroLevelDetector(instrument_token=instrument_token)
        
        # Parse analysis date
        if analysis_date:
            try:
                target_date = datetime.strptime(analysis_date, '%Y-%m-%d').date()
            except:
                target_date = date.today()
        elif ANALYSIS_DATE:
            target_date = datetime.strptime(ANALYSIS_DATE, '%Y-%m-%d').date()
        else:
            target_date = date.today()
        
        # Detect critical levels
        levels = detector.detect_critical_levels(
            start_time=start_time,
            end_time=end_time,
            analysis_date=target_date
        )
        
        return levels
    except Exception as e:
        logging.error(f"Error detecting micro levels: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/api/export_data")
async def api_export_data(
    table_name: str = Form(...),
    from_date: str = Form(None),
    to_date: str = Form(None),
    columns: str = Form(None),
    export_format: str = Form("csv")
):
    """API endpoint to export data from any table with date range and column filtering"""
    try:
        import pandas as pd
        from datetime import datetime, date
        from io import BytesIO
        from fastapi.responses import StreamingResponse
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Validate table name (security: whitelist approach)
        allowed_tables = [
            'ticks', 'futures_ticks', 'options_ticks', 'holdings', 'mf_holdings',
            'positions', 'orders', 'trades', 'market_structure', 'rt_intraday_price',
            'iv_history', 'market_depth', 'futures_tick_depth', 'raw_ticks', 'bars',
            'profile', 'instruments', 'prophet_predictions', 'swing_trade_suggestions'
        ]
        
        if table_name not in allowed_tables:
            return {"success": False, "error": f"Table '{table_name}' is not allowed for export"}
        
        # Build SELECT query
        if columns and columns.strip():
            # User specified columns
            column_list = [col.strip() for col in columns.split(',')]
            # Sanitize column names (remove any SQL injection attempts)
            column_list = [col for col in column_list if col.replace('_', '').replace('.', '').isalnum()]
            if not column_list:
                column_list = ['*']
            columns_str = ', '.join(column_list)
        else:
            columns_str = '*'
        
        # Build WHERE clause for date filtering
        where_clauses = []
        params = []
        
        # Find date columns in the table
        date_columns = []
        if table_name in ['ticks', 'futures_ticks', 'options_ticks']:
            date_columns = ['run_date', 'timestamp']
        elif table_name in ['holdings', 'mf_holdings', 'positions', 'orders', 'trades']:
            date_columns = ['run_date', 'timestamp', 'order_timestamp', 'trade_timestamp']
        elif table_name in ['market_structure', 'sessions']:
            date_columns = ['session_date', 'run_date']
        elif table_name == 'rt_intraday_price':
            date_columns = ['price_date', 'insert_date']
        elif table_name == 'iv_history':
            date_columns = ['price_date']
        else:
            date_columns = ['run_date', 'timestamp']
        
        # Apply date filters if provided
        if from_date:
            try:
                from_date_obj = datetime.strptime(from_date, '%Y-%m-%d').date()
                # Try to filter on first available date column
                if date_columns:
                    where_clauses.append(f"{date_columns[0]} >= %s")
                    params.append(from_date_obj)
            except:
                pass
        
        if to_date:
            try:
                to_date_obj = datetime.strptime(to_date, '%Y-%m-%d').date()
                if date_columns:
                    where_clauses.append(f"{date_columns[0]} <= %s")
                    params.append(to_date_obj)
            except:
                pass
        
        where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Execute query
        query = f"SELECT {columns_str} FROM my_schema.{table_name}{where_clause}"
        cursor.execute(query, params)
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Fetch all rows
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {"success": False, "error": "No data found for the specified criteria"}
        
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=column_names)
        
        # Generate file based on format
        if export_format == "csv":
            output = BytesIO()
            df.to_csv(output, index=False, encoding='utf-8')
            output.seek(0)
            return StreamingResponse(
                output,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={table_name}_{from_date or 'all'}_{to_date or 'all'}.csv"}
            )
        
        elif export_format == "excel":
            try:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=table_name, index=False)
                output.seek(0)
                return StreamingResponse(
                    output,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename={table_name}_{from_date or 'all'}_{to_date or 'all'}.xlsx"}
                )
            except Exception as e:
                logging.error(f"Error generating Excel: {e}")
                return {"success": False, "error": f"Excel generation failed: {str(e)}"}
        
        elif export_format == "json":
            output = BytesIO()
            df.to_json(output, orient='records', date_format='iso', indent=2)
            output.seek(0)
            return StreamingResponse(
                output,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={table_name}_{from_date or 'all'}_{to_date or 'all'}.json"}
            )
        
        else:
            return {"success": False, "error": f"Unsupported format: {export_format}"}
            
    except Exception as e:
        logging.error(f"Error exporting data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/api/import_data")
async def api_import_data(
    table_name: str = Form(...),
    file: UploadFile = File(...)
):
    """API endpoint to import CSV data into any table with conflict handling (skip duplicates)"""
    try:
        import pandas as pd
        import csv
        from io import StringIO
        from psycopg2.extras import execute_batch
        
        # Validate table name (security: whitelist approach)
        allowed_tables = [
            'ticks', 'futures_ticks', 'options_ticks', 'holdings', 'mf_holdings',
            'positions', 'orders', 'trades', 'market_structure', 'rt_intraday_price',
            'iv_history', 'market_depth', 'futures_tick_depth', 'raw_ticks', 'bars',
            'profile', 'instruments'
        ]
        
        if table_name not in allowed_tables:
            return {"success": False, "error": f"Table '{table_name}' is not allowed for import"}
        
        # Read CSV file
        contents = await file.read()
        csv_content = contents.decode('utf-8')
        
        # Parse CSV
        df = pd.read_csv(StringIO(csv_content))
        
        if df.empty:
            return {"success": False, "error": "CSV file is empty"}
        
        # Get table primary key/unique constraints to determine conflict columns
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get primary key columns for conflict handling
        cursor.execute("""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = 'my_schema.%s'::regclass
            AND i.indisprimary
        """ % table_name)
        pk_columns = [row[0] for row in cursor.fetchall()]
        
        # Also check unique constraints
        cursor.execute("""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = 'my_schema.%s'::regclass
            AND i.indisunique
            AND NOT i.indisprimary
        """ % table_name)
        unique_columns = [row[0] for row in cursor.fetchall()]
        
        # Combine primary key and unique columns for conflict handling
        conflict_columns = pk_columns + unique_columns
        
        if not conflict_columns:
            # If no primary key, use all columns for conflict (less ideal but works)
            # For now, we'll try to insert all and let PostgreSQL handle duplicates
            conflict_columns = list(df.columns)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Convert DataFrame rows to list of tuples
        rows_to_insert = [tuple(row) for row in df.values]
        
        # Build INSERT query
        columns_str = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(df.columns))
        
        insert_query = f"""
            INSERT INTO my_schema.{table_name} ({columns_str})
            VALUES ({placeholders})
        """
        
        # Add ON CONFLICT clause if we have conflict columns
        if conflict_columns and len(conflict_columns) > 0:
            # Find intersection of conflict columns and CSV columns
            available_conflict_cols = [col for col in conflict_columns if col in df.columns]
            
            if available_conflict_cols:
                conflict_cols_str = ', '.join(available_conflict_cols)
                insert_query += f" ON CONFLICT ({conflict_cols_str}) DO NOTHING"
        
        # Execute batch insert
        total_rows = len(rows_to_insert)
        execute_batch(cursor, insert_query, rows_to_insert)
        conn.commit()
        
        # Count how many were actually inserted (not skipped due to conflicts)
        rows_inserted = cursor.rowcount if cursor.rowcount >= 0 else 0
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Import completed: {rows_inserted} rows inserted, {total_rows - rows_inserted} rows skipped (conflicts)",
            "total_rows": total_rows,
            "rows_inserted": rows_inserted,
            "rows_skipped": total_rows - rows_inserted,
            "table_name": table_name
        }
        
    except Exception as e:
        logging.error(f"Error importing data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/api/refresh_stock_prices")
async def api_refresh_stock_prices():
    """API endpoint to refresh stock price data from Yahoo Finance"""
    try:
        from kite.InsertOHLC import refresh_stock_prices
        
        # Get database config from environment or use defaults
        db_config = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'database': os.getenv('DB_NAME', 'mydb'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        
        # Call the refresh function
        result = refresh_stock_prices(db_config)
        
        return result
        
    except Exception as e:
        logging.error(f"Error refreshing stock prices: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/api/add_new_stock")
async def api_add_new_stock(request: Request):
    """API endpoint to add a new stock to master_scrips and fetch 6 months of historical data"""
    try:
        import pandas as pd
        import yfinance as yf
        from datetime import datetime, timedelta
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        
        # Parse request body
        body = await request.json()
        symbol = (body.get('symbol') or '').strip().upper()
        yahoo_code = (body.get('yahoo_code') or '').strip()
        country = (body.get('country') or 'IN').strip() or 'IN'
        sector_code = (body.get('sector_code') or '').strip() or None
        
        # Validation
        if not symbol:
            return {"success": False, "error": "Stock symbol is required"}
        
        if not yahoo_code:
            return {"success": False, "error": "Yahoo Finance code is required"}
        
        # Get database config
        db_config = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'database': os.getenv('DB_NAME', 'mydb'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(connection_string)
        db_connection = engine.connect()
        session = sessionmaker(bind=engine)()
        
        result = {
            'success': False,
            'message': '',
            'records_inserted': 0,
            'date_range': None,
            'error': None
        }
        
        try:
            # Check if stock already exists
            check_query = text("""
                SELECT scrip_id, yahoo_code FROM my_schema.master_scrips 
                WHERE scrip_id = :symbol AND scrip_country = :country
            """)
            existing = db_connection.execute(check_query, {'symbol': symbol, 'country': country}).fetchone()
            
            stock_exists = existing is not None
            needs_data_fetch = False
            
            if stock_exists:
                # Check if there's any data in rt_intraday_price
                check_data_query = text("""
                    SELECT COUNT(*) as count FROM my_schema.rt_intraday_price 
                    WHERE scrip_id = :symbol AND country = :country
                """)
                data_count = db_connection.execute(check_data_query, {'symbol': symbol, 'country': country}).fetchone()
                
                if data_count and data_count[0] > 0:
                    # Stock exists and has data - return error
                    db_connection.close()
                    engine.dispose()
                    return {"success": False, "error": f"Stock {symbol} already exists in the database with historical data"}
                
                # Stock exists but no data - we'll update yahoo_code and fetch data
                needs_data_fetch = True
                existing_yahoo_code = existing[1] if existing else None
                
                # Update yahoo_code if it's different
                if existing_yahoo_code != yahoo_code:
                    update_query = text("""
                        UPDATE my_schema.master_scrips 
                        SET yahoo_code = :yahoo_code, updated_at = NOW()
                        WHERE scrip_id = :symbol AND scrip_country = :country
                    """)
                    db_connection.execute(update_query, {'yahoo_code': yahoo_code, 'symbol': symbol, 'country': country})
                    db_connection.commit()
                    logging.info(f"Updated yahoo_code for {symbol} from {existing_yahoo_code} to {yahoo_code}")
                else:
                    # Use existing yahoo_code
                    yahoo_code = existing_yahoo_code or yahoo_code
                    logging.info(f"Stock {symbol} exists with same yahoo_code {yahoo_code}, fetching historical data")
            else:
                # Add new stock to master_scrips
                insert_query = text("""
                    INSERT INTO my_schema.master_scrips 
                    (scrip_id, yahoo_code, scrip_country, sector_code, created_at, updated_at)
                    VALUES (:symbol, :yahoo_code, :country, :sector_code, NOW(), NOW())
                """)
                
                params = {
                    'symbol': symbol,
                    'yahoo_code': yahoo_code,
                    'country': country,
                    'sector_code': sector_code
                }
                
                db_connection.execute(insert_query, params)
                db_connection.commit()
                logging.info(f"Successfully added {symbol} to master_scrips")
                needs_data_fetch = True
            
            # Only fetch data if needed (new stock or existing stock without data)
            if not needs_data_fetch:
                db_connection.close()
                engine.dispose()
                return {
                    "success": False,
                    "error": f"Stock {symbol} already exists in the database with historical data"
                }
            
            # Calculate date range for 6 months
            end_date = datetime.now().date() + timedelta(days=1)  # Today + 1 day (exclusive end)
            start_date = end_date - timedelta(days=180)  # Approximately 6 months (180 days)
            
            result['date_range'] = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
            
            # Fetch historical data from Yahoo Finance
            logging.info(f"Fetching 6 months of historical data for {symbol} ({yahoo_code}) from {start_date} to {end_date}")
            quote = yf.download(yahoo_code, start=start_date, end=end_date, progress=False)
            
            if quote.empty:
                db_connection.close()
                engine.dispose()
                return {
                    "success": False,
                    "error": f"No historical data found for {yahoo_code}. Please verify the Yahoo Finance code is correct."
                }
            
            records_inserted = 0
            
            # Insert historical data into rt_intraday_price
            for date, dailyrow in quote.iterrows():
                insert_price_query = text("""
                    INSERT INTO my_schema.rt_intraday_price 
                    (scrip_id, price_close, price_high, price_low, price_open, price_date, country, volume) 
                    VALUES (:scrip_id, :close, :high, :low, :open, :date, :country, :volume)
                    ON CONFLICT (scrip_id, price_date) 
                    DO UPDATE SET 
                        price_close = EXCLUDED.price_close,
                        price_high = EXCLUDED.price_high,
                        price_low = EXCLUDED.price_low,
                        price_open = EXCLUDED.price_open,
                        country = EXCLUDED.country,
                        volume = EXCLUDED.volume,
                        created_at = CURRENT_TIMESTAMP
                """)
                
                try:
                    # Extract values from yfinance data
                    # yfinance returns DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
                    # Access by column name (dailyrow is a pandas Series with index as column names)
                    try:
                        open_price = float(dailyrow['Open']) if 'Open' in dailyrow.index and not pd.isna(dailyrow['Open']) else None
                        high_price = float(dailyrow['High']) if 'High' in dailyrow.index and not pd.isna(dailyrow['High']) else None
                        low_price = float(dailyrow['Low']) if 'Low' in dailyrow.index and not pd.isna(dailyrow['Low']) else None
                        close_price = float(dailyrow['Close']) if 'Close' in dailyrow.index and not pd.isna(dailyrow['Close']) else None
                        volume_value = int(dailyrow['Volume']) if 'Volume' in dailyrow.index and not pd.isna(dailyrow['Volume']) else 0
                    except (KeyError, IndexError):
                        # Fallback to positional access if column names not available
                        # yfinance returns: Open, High, Low, Close, Adj Close, Volume
                        # So: values[0]=Open, values[1]=High, values[2]=Low, values[3]=Close, values[5]=Volume
                        open_price = float(dailyrow.values[0]) if len(dailyrow.values) > 0 and not pd.isna(dailyrow.values[0]) else None
                        high_price = float(dailyrow.values[1]) if len(dailyrow.values) > 1 and not pd.isna(dailyrow.values[1]) else None
                        low_price = float(dailyrow.values[2]) if len(dailyrow.values) > 2 and not pd.isna(dailyrow.values[2]) else None
                        close_price = float(dailyrow.values[3]) if len(dailyrow.values) > 3 and not pd.isna(dailyrow.values[3]) else None
                        volume_value = int(dailyrow.values[5]) if len(dailyrow.values) > 5 and not pd.isna(dailyrow.values[5]) else 0
                    
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
                    
                    db_connection.execute(
                        insert_price_query,
                        {
                            'scrip_id': symbol,
                            'close': close_price,
                            'high': high_price,
                            'low': low_price,
                            'open': open_price,
                            'date': date_str,
                            'country': country,
                            'volume': volume_value
                        }
                    )
                    records_inserted += 1
                    
                except Exception as e:
                    logging.warning(f"Error inserting price data for {symbol} on {date}: {str(e)}")
                    continue
            
            db_connection.commit()
            
            result['success'] = True
            # Determine if this was a new stock or existing stock update
            if stock_exists:
                result['message'] = f"Stock {symbol} already existed in database. Updated yahoo_code and fetched {records_inserted} historical records"
            else:
                result['message'] = f"Successfully added {symbol} to database and fetched {records_inserted} historical records"
            result['records_inserted'] = records_inserted
            
            logging.info(f"Successfully {'updated' if stock_exists else 'added'} {symbol} with {records_inserted} historical records")
            
        except Exception as e:
            error_msg = f"Error adding stock {symbol}: {str(e)}"
            logging.error(error_msg)
            import traceback
            logging.error(traceback.format_exc())
            result['error'] = error_msg
            result['success'] = False
            result['message'] = error_msg
            
        finally:
            db_connection.close()
            engine.dispose()
        
        return result
        
    except Exception as e:
        logging.error(f"Error in add_new_stock API: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.get("/api/swing_trades")
async def api_swing_trades(
    min_gain: float = Query(10.0, description="Minimum target gain %"),
    max_gain: float = Query(20.0, description="Maximum target gain %"),
    min_confidence: float = Query(70.0, description="Minimum confidence score"),
    pattern_type: str = Query(None, description="Filter by pattern type"),
    limit: int = Query(20, description="Number of results"),
    scan_limit: int = Query(50, description="Limit number of stocks to scan (default: 50)"),
    force_refresh: bool = Query(False, description="Force refresh by generating new recommendations")
):
    """API endpoint to get swing trade recommendations for stocks
    
    By default, reads from database (refreshed every 30 minutes via cron).
    Set force_refresh=True to generate new recommendations on-demand.
    """
    try:
        from stocks.SwingTradeScanner import SwingTradeScanner
        from datetime import date
        import json
        
        today = date.today()
        
        # Try to read from database first (unless force_refresh is True)
        if not force_refresh:
            try:
                conn = get_db_connection()
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Build query to get latest recommendations for today
                where_clauses = ["run_date = %s", "status = 'ACTIVE'"]
                params = [today]
                
                # Apply filtering criteria
                if min_gain:
                    where_clauses.append("potential_gain_pct >= %s")
                    params.append(min_gain)
                if max_gain:
                    where_clauses.append("potential_gain_pct <= %s")
                    params.append(max_gain)
                if min_confidence:
                    where_clauses.append("confidence_score >= %s")
                    params.append(min_confidence)
                if pattern_type:
                    where_clauses.append("pattern_type = %s")
                    params.append(pattern_type)
                
                sql = f"""
                    SELECT 
                        scrip_id, instrument_token, pattern_type, direction,
                        entry_price, target_price, stop_loss,
                        potential_gain_pct, risk_reward_ratio, confidence_score,
                        holding_period_days, current_price,
                        sma_20, sma_50, sma_200, rsi_14, macd, macd_signal, atr_14,
                        volume_trend, support_level, resistance_level, rationale,
                        technical_context, diagnostics, filtering_criteria,
                        analysis_date, generated_at
                    FROM my_schema.swing_trade_suggestions
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY confidence_score DESC
                    LIMIT %s
                """
                params.append(limit if limit > 0 else 1000)
                
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
                
                if rows and len(rows) > 0:
                    # Convert database rows to recommendation format
                    recommendations = []
                    for row in rows:
                        rec = {
                            'scrip_id': row.get('scrip_id'),
                            'instrument_token': row.get('instrument_token'),
                            'pattern_type': row.get('pattern_type'),
                            'direction': row.get('direction', 'BUY'),
                            'entry_price': float(row.get('entry_price', 0)) if row.get('entry_price') else None,
                            'target_price': float(row.get('target_price', 0)) if row.get('target_price') else None,
                            'stop_loss': float(row.get('stop_loss', 0)) if row.get('stop_loss') else None,
                            'potential_gain_pct': float(row.get('potential_gain_pct', 0)) if row.get('potential_gain_pct') else None,
                            'risk_reward_ratio': float(row.get('risk_reward_ratio', 0)) if row.get('risk_reward_ratio') else None,
                            'confidence_score': float(row.get('confidence_score', 0)) if row.get('confidence_score') else None,
                            'holding_period_days': int(row.get('holding_period_days', 0)) if row.get('holding_period_days') else None,
                            'current_price': float(row.get('current_price', 0)) if row.get('current_price') else None,
                            'sma_20': float(row.get('sma_20', 0)) if row.get('sma_20') else None,
                            'sma_50': float(row.get('sma_50', 0)) if row.get('sma_50') else None,
                            'sma_200': float(row.get('sma_200', 0)) if row.get('sma_200') else None,
                            'rsi_14': float(row.get('rsi_14', 0)) if row.get('rsi_14') else None,
                            'macd': float(row.get('macd', 0)) if row.get('macd') else None,
                            'macd_signal': float(row.get('macd_signal', 0)) if row.get('macd_signal') else None,
                            'atr_14': float(row.get('atr_14', 0)) if row.get('atr_14') else None,
                            'volume_trend': row.get('volume_trend'),
                            'support_level': float(row.get('support_level', 0)) if row.get('support_level') else None,
                            'resistance_level': float(row.get('resistance_level', 0)) if row.get('resistance_level') else None,
                            'rationale': row.get('rationale'),
                        }
                        
                        # Parse JSONB fields
                        if row.get('technical_context'):
                            try:
                                if isinstance(row['technical_context'], str):
                                    rec['technical_context'] = json.loads(row['technical_context'])
                                else:
                                    rec['technical_context'] = row['technical_context']
                            except:
                                rec['technical_context'] = {}
                        
                        if row.get('diagnostics'):
                            try:
                                if isinstance(row['diagnostics'], str):
                                    rec['diagnostics'] = json.loads(row['diagnostics'])
                                else:
                                    rec['diagnostics'] = row['diagnostics']
                            except:
                                rec['diagnostics'] = {}
                        
                        if row.get('filtering_criteria'):
                            try:
                                if isinstance(row['filtering_criteria'], str):
                                    rec['filtering_criteria'] = json.loads(row['filtering_criteria'])
                                else:
                                    rec['filtering_criteria'] = row['filtering_criteria']
                            except:
                                rec['filtering_criteria'] = {}
                        
                        recommendations.append(rec)
                    
                    # Get Prophet predictions and enrich recommendations
                    try:
                        cursor.execute("""
                            SELECT scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days, run_date
                            FROM my_schema.prophet_predictions
                            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE' AND prediction_days = 30)
                            AND prediction_days = 30
                            AND status = 'ACTIVE'
                        """)
                        
                        predictions_rows = cursor.fetchall()
                        
                        if not predictions_rows:
                            cursor.execute("""
                                SELECT scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days, run_date
                                FROM my_schema.prophet_predictions pp1
                                WHERE status = 'ACTIVE'
                                AND run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE')
                            """)
                            predictions_rows = cursor.fetchall()
                        
                        predictions_map = {}
                        if predictions_rows:
                            for row in predictions_rows:
                                scrip_id = row['scrip_id'].upper()
                                if scrip_id not in predictions_map:
                                    predictions_map[scrip_id] = dict(row)
                        
                        # Add predictions to recommendations
                        for rec in recommendations:
                            scrip_id = rec.get('scrip_id')
                            if scrip_id:
                                scrip_id_upper = scrip_id.upper()
                                if scrip_id_upper in predictions_map:
                                    pred = predictions_map[scrip_id_upper]
                                    try:
                                        pred_pct = pred.get('predicted_price_change_pct')
                                        if pred_pct is not None and not (isinstance(pred_pct, float) and (pred_pct != pred_pct)):
                                            rec['prophet_prediction_pct'] = float(pred_pct)
                                        else:
                                            rec['prophet_prediction_pct'] = None
                                    except:
                                        rec['prophet_prediction_pct'] = None
                                    
                                    try:
                                        pred_conf = pred.get('prediction_confidence')
                                        if pred_conf is not None and not (isinstance(pred_conf, float) and (pred_conf != pred_conf)):
                                            rec['prophet_confidence'] = float(pred_conf)
                                        else:
                                            rec['prophet_confidence'] = None
                                    except:
                                        rec['prophet_confidence'] = None
                                    
                                    pred_days = pred.get('prediction_days')
                                    rec['prediction_days'] = int(pred_days) if pred_days is not None else None
                                else:
                                    rec['prophet_prediction_pct'] = None
                                    rec['prophet_confidence'] = None
                                    rec['prediction_days'] = None
                            else:
                                rec['prophet_prediction_pct'] = None
                                rec['prophet_confidence'] = None
                                rec['prediction_days'] = None
                    except Exception as pred_err:
                        logging.debug(f"Error enriching with Prophet predictions: {pred_err}")
                        for rec in recommendations:
                            rec['prophet_prediction_pct'] = None
                            rec['prophet_confidence'] = None
                            rec['prediction_days'] = None
                    
                    cursor.close()
                    conn.close()
                    
                    # Return cached results
                    return cached_json_response({
                        "success": True,
                        "recommendations": recommendations[:limit] if limit > 0 else recommendations,
                        "total_found": len(recommendations),
                        "analysis_date": str(today),
                        "source": "database",
                        "filtering_criteria": {
                            "min_gain": min_gain,
                            "max_gain": max_gain,
                            "min_confidence": min_confidence,
                            "pattern_type": pattern_type
                        }
                    }, "/api/swing_trades")
                
                cursor.close()
                conn.close()
            except Exception as db_err:
                logging.warning(f"Error reading from database, will generate new recommendations: {db_err}")
                # Continue to generate new recommendations
        
        # If database read failed or force_refresh=True, generate new recommendations
        scanner = SwingTradeScanner(
            min_gain=min_gain,
            max_gain=max_gain,
            min_confidence=min_confidence
        )
        
        # Scan stocks with limit (default: 50 to keep scan time reasonable)
        # If scan_limit is None, use 50 as default
        actual_scan_limit = scan_limit if scan_limit is not None else 50
        recommendations = scanner.scan_all_stocks(limit=actual_scan_limit)
        
        # Also scan Nifty50 and add to recommendations
        nifty_recommendations = scanner.scan_nifty()
        if nifty_recommendations:
            recommendations.extend(nifty_recommendations)
        
        # Filter by pattern type if provided
        if pattern_type:
            recommendations = [r for r in recommendations if r.get('pattern_type') == pattern_type]
        
        # Get Prophet predictions from latest run_date for all stocks
        from stocks.ProphetPricePredictor import ProphetPricePredictor
        predictor = ProphetPricePredictor()
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        try:
            # First, try to get 30-day predictions (most common)
            cursor.execute("""
                SELECT scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days, run_date, prediction_details
                FROM my_schema.prophet_predictions
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE' AND prediction_days = 30)
                AND prediction_days = 30
                AND status = 'ACTIVE'
            """)
            
            predictions_rows = cursor.fetchall()
            
            # If no 30-day predictions found, get the latest predictions regardless of prediction_days
            if not predictions_rows:
                logging.warning("No 30-day Prophet predictions found, trying to get latest predictions for any prediction_days")
                cursor.execute("""
                    SELECT scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days, run_date, prediction_details
                    FROM my_schema.prophet_predictions pp1
                    WHERE status = 'ACTIVE'
                    AND run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE')
                """)
                predictions_rows = cursor.fetchall()
                if predictions_rows:
                    logging.info(f"Found {len(predictions_rows)} Prophet predictions with latest run_date (not necessarily 30 days)")
            
            # Build predictions map, handling duplicate scrip_ids by keeping the most recent or 30-day version
            predictions_map = {}
            if predictions_rows:
                for row in predictions_rows:
                    scrip_id = row['scrip_id'].upper()
                    if scrip_id not in predictions_map:
                        predictions_map[scrip_id] = dict(row)
                    else:
                        # Prefer 30-day predictions, or most recent if both are same prediction_days
                        existing = predictions_map[scrip_id]
                        existing_days = existing.get('prediction_days', 0)
                        row_days = row.get('prediction_days', 0)
                        
                        if row_days == 30 or (existing_days != 30 and row_days > existing_days):
                            predictions_map[scrip_id] = dict(row)
            
            logging.info(f"Found {len(predictions_map)} Prophet predictions to match with {len(recommendations)} recommendations")
            
            # Debug: Get all Prophet prediction scrip_ids from database for comparison
            cursor.execute("""
                SELECT DISTINCT scrip_id, prediction_days, run_date, COUNT(*) as count
                FROM my_schema.prophet_predictions
                WHERE status = 'ACTIVE'
                GROUP BY scrip_id, prediction_days, run_date
                ORDER BY run_date DESC, prediction_days DESC
                LIMIT 20
            """)
            all_preds_sample = cursor.fetchall()
            if all_preds_sample:
                logging.info(f"Sample Prophet predictions in DB: {[(r['scrip_id'], r['prediction_days'], r['run_date']) for r in all_preds_sample[:10]]}")
            
            # Debug: Log sample prediction scrip_ids
            if predictions_map:
                sample_pred_ids = list(predictions_map.keys())[:10]
                logging.info(f"Prophet prediction scrip_ids to match: {sample_pred_ids}")
            
            # Debug: Log sample recommendation scrip_ids
            if recommendations:
                sample_rec_ids = [rec.get('scrip_id', 'N/A') for rec in recommendations[:10]]
                logging.info(f"Recommendation scrip_ids to match: {sample_rec_ids}")
                
                # Check for any common scrip_ids
                rec_ids_upper = set([rec.get('scrip_id', '').upper() for rec in recommendations if rec.get('scrip_id')])
                pred_ids_upper = set(predictions_map.keys())
                common_ids = rec_ids_upper.intersection(pred_ids_upper)
                if common_ids:
                    logging.info(f"✓ Found {len(common_ids)} common scrip_ids: {sorted(list(common_ids))[:10]}")
                else:
                    logging.warning(f"⚠ No common scrip_ids found! Recommendations: {sorted(list(rec_ids_upper))[:5]}, Predictions: {sorted(list(pred_ids_upper))[:5]}")
            
            # Track matching stats
            matched_count = 0
            unmatched_scrip_ids = set()
            matched_scrip_ids = []
            
            # Add predictions to recommendations (case-insensitive matching)
            for rec in recommendations:
                scrip_id = rec.get('scrip_id')
                if scrip_id:
                    scrip_id_upper = scrip_id.upper()
                    if scrip_id_upper in predictions_map:
                        pred = predictions_map[scrip_id_upper]
                        # Convert to float, handling None and NaN cases
                        pred_pct = pred.get('predicted_price_change_pct')
                        pred_conf = pred.get('prediction_confidence')
                        
                        try:
                            if pred_pct is not None and not (isinstance(pred_pct, float) and (pred_pct != pred_pct)):  # Check for NaN
                                rec['prophet_prediction_pct'] = float(pred_pct)
                            else:
                                rec['prophet_prediction_pct'] = None
                        except (ValueError, TypeError):
                            rec['prophet_prediction_pct'] = None
                        
                        try:
                            if pred_conf is not None and not (isinstance(pred_conf, float) and (pred_conf != pred_conf)):  # Check for NaN
                                rec['prophet_confidence'] = float(pred_conf)
                            else:
                                rec['prophet_confidence'] = None
                        except (ValueError, TypeError):
                            rec['prophet_confidence'] = None
                        
                        # Add prediction_days if available
                        pred_days = pred.get('prediction_days')
                        if pred_days is not None:
                            try:
                                rec['prediction_days'] = int(pred_days)
                            except (ValueError, TypeError):
                                rec['prediction_days'] = None
                        else:
                            rec['prediction_days'] = None
                        
                        # Extract cv_metrics from prediction_details
                        prediction_details = pred.get('prediction_details')
                        if prediction_details:
                            try:
                                # PostgreSQL JSONB might return as dict or string depending on psycopg2 version
                                if isinstance(prediction_details, str):
                                    details = json.loads(prediction_details)
                                elif isinstance(prediction_details, dict):
                                    details = prediction_details
                                else:
                                    # Try to convert to string first
                                    details = json.loads(str(prediction_details))
                                
                                cv_metrics = details.get('cv_metrics')
                                if cv_metrics and isinstance(cv_metrics, dict):
                                    # Handle None, infinity, or invalid values
                                    mape_value = cv_metrics.get('mape')
                                    rmse_value = cv_metrics.get('rmse')
                                    
                                    # Convert infinity strings or infinity values to None
                                    if mape_value is None or mape_value == float('inf') or mape_value == float('-inf') or (isinstance(mape_value, str) and mape_value.lower() in ['inf', 'infinity', 'nan']):
                                        rec['prophet_cv_mape'] = None
                                    else:
                                        try:
                                            rec['prophet_cv_mape'] = float(mape_value) if mape_value is not None else None
                                        except (ValueError, TypeError):
                                            rec['prophet_cv_mape'] = None
                                    
                                    if rmse_value is None or rmse_value == float('inf') or rmse_value == float('-inf') or (isinstance(rmse_value, str) and rmse_value.lower() in ['inf', 'infinity', 'nan']):
                                        rec['prophet_cv_rmse'] = None
                                    else:
                                        try:
                                            rec['prophet_cv_rmse'] = float(rmse_value) if rmse_value is not None else None
                                        except (ValueError, TypeError):
                                            rec['prophet_cv_rmse'] = None
                                    
                                    logging.debug(f"Extracted cv_metrics for {scrip_id}: MAPE={rec['prophet_cv_mape']}, RMSE={rec['prophet_cv_rmse']}")
                                else:
                                    rec['prophet_cv_mape'] = None
                                    rec['prophet_cv_rmse'] = None
                                    logging.debug(f"No cv_metrics found in prediction_details for {scrip_id}, cv_metrics={cv_metrics}")
                            except (json.JSONDecodeError, Exception) as e:
                                logging.warning(f"Error parsing prediction_details for {scrip_id}: {e}, type={type(prediction_details)}")
                                rec['prophet_cv_mape'] = None
                                rec['prophet_cv_rmse'] = None
                        else:
                            rec['prophet_cv_mape'] = None
                            rec['prophet_cv_rmse'] = None
                            logging.debug(f"No prediction_details found for {scrip_id}")
                        
                        if rec['prophet_prediction_pct'] is not None:
                            matched_count += 1
                            matched_scrip_ids.append(scrip_id)
                            logging.debug(f"Matched Prophet prediction for {scrip_id}: {rec['prophet_prediction_pct']:.2f}%")
                    else:
                        rec['prophet_prediction_pct'] = None
                        rec['prophet_confidence'] = None
                        rec['prediction_days'] = None
                        rec['prophet_cv_mape'] = None
                        rec['prophet_cv_rmse'] = None
                        unmatched_scrip_ids.add(scrip_id)
                        logging.debug(f"No Prophet prediction found for {scrip_id} (looking for: {scrip_id_upper})")
                else:
                    rec['prophet_prediction_pct'] = None
                    rec['prophet_confidence'] = None
                    rec['prediction_days'] = None
                    rec['prophet_cv_mape'] = None
                    rec['prophet_cv_rmse'] = None
            
            if matched_count > 0:
                logging.info(f"✓ Matched {matched_count} recommendations with Prophet predictions")
                logging.info(f"Matched scrip_ids: {matched_scrip_ids[:10]}{'...' if len(matched_scrip_ids) > 10 else ''}")
            else:
                logging.warning(f"⚠ No Prophet predictions matched! Check if predictions exist in database.")
                logging.warning(f"  - Prophet predictions available: {len(predictions_map)}")
                logging.warning(f"  - Recommendations to match: {len(recommendations)}")
                if predictions_map and recommendations:
                    logging.warning(f"  - Prophet prediction scrip_ids: {sorted(list(predictions_map.keys()))[:10]}")
                    logging.warning(f"  - Recommendation scrip_ids: {sorted([r.get('scrip_id', 'N/A').upper() for r in recommendations[:10]])}")
            
            if unmatched_scrip_ids:
                logging.info(f"Unmatched scrip_ids (no Prophet prediction found): {sorted(list(unmatched_scrip_ids))[:10]}{'...' if len(unmatched_scrip_ids) > 10 else ''}")
                
        except Exception as e:
            logging.error(f"Error fetching Prophet predictions: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Still continue without predictions
            for rec in recommendations:
                rec['prophet_prediction_pct'] = None
                rec['prophet_confidence'] = None
                rec['prediction_days'] = None
                rec['prophet_cv_mape'] = None
                rec['prophet_cv_rmse'] = None
        finally:
            cursor.close()
            conn.close()
        
        # Sort by confidence score (highest first)
        recommendations.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
        
        # Limit results
        if limit > 0:
            recommendations = recommendations[:limit]
        
        # Save to database
        analysis_date = date.today()
        scanner.save_recommendations(recommendations, analysis_date)
        
        return cached_json_response({
            "success": True,
            "recommendations": recommendations,
            "total_found": len(recommendations),
            "analysis_date": str(analysis_date),
            "source": "generated",
            "filtering_criteria": {
                "min_gain": min_gain,
                "max_gain": max_gain,
                "min_confidence": min_confidence,
                "pattern_type": pattern_type
            }
        }, "/api/swing_trades")
        
    except Exception as e:
        logging.error(f"Error generating swing trade recommendations: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.get("/api/swing_trades_nifty")
async def api_swing_trades_nifty(
    min_gain: float = Query(10.0, description="Minimum target gain %"),
    max_gain: float = Query(20.0, description="Maximum target gain %"),
    min_confidence: float = Query(70.0, description="Minimum confidence score")
):
    """API endpoint to get swing trade recommendations for Nifty"""
    try:
        from stocks.SwingTradeScanner import SwingTradeScanner
        from datetime import date
        
        scanner = SwingTradeScanner(
            min_gain=min_gain,
            max_gain=max_gain,
            min_confidence=min_confidence
        )
        
        # Scan Nifty
        recommendations = scanner.scan_nifty()
        
        # Save to database
        analysis_date = date.today()
        scanner.save_recommendations(recommendations, analysis_date)
        
        return {
            "success": True,
            "recommendations": recommendations,
            "total_found": len(recommendations),
            "analysis_date": str(analysis_date),
            "filtering_criteria": {
                "min_gain": min_gain,
                "max_gain": max_gain,
                "min_confidence": min_confidence
            }
        }
        
    except Exception as e:
        logging.error(f"Error generating Nifty swing trade recommendations: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.get("/api/prophet_predictions/generate")
async def api_generate_prophet_predictions(
    prediction_days: int = Query(30, description="Number of days to predict ahead (default: 30, supports 30, 60, 90, 180, etc.)"),
    limit: int = Query(None, description="Limit number of stocks to process (default: all)"),
    force: bool = Query(False, description="Force regeneration even if predictions exist for today")
):
    """API endpoint to generate Prophet price predictions for all stocks (runs once per day)"""
    try:
        from stocks.ProphetPricePredictor import ProphetPricePredictor
        from datetime import date
        
        predictor = ProphetPricePredictor(prediction_days=prediction_days)
        run_date = date.today()
        
        # Check if predictions already exist for today and this prediction_days (unless force=True)
        if not force and predictor.check_predictions_exist_for_date(run_date, prediction_days=prediction_days):
            return {
                "success": False,
                "error": f"Predictions already generated for today with {prediction_days} days. Set force=true to regenerate.",
                "run_date": str(run_date),
                "prediction_days": prediction_days
            }
        
        # Generate predictions
        logging.info(f"Starting Prophet prediction generation for run_date={run_date}, prediction_days={prediction_days}, limit={limit}")
        predictions = predictor.predict_all_stocks(limit=limit, prediction_days=prediction_days)
        
        logging.info(f"Prophet prediction generation completed: {len(predictions)} predictions generated")
        
        if not predictions:
            error_msg = "No predictions generated. This could be due to: insufficient data (need at least 60 days), Prophet model errors, or data quality issues. Check application logs for details."
            logging.warning(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "run_date": str(run_date),
                "prediction_days": prediction_days
            }
        
        # Save predictions
        save_success = predictor.save_predictions(predictions, run_date, prediction_days=prediction_days)
        
        if not save_success:
            return {
                "success": False,
                "error": "Failed to save predictions to database",
                "predictions_generated": len(predictions),
                "run_date": str(run_date)
            }
        
        return {
            "success": True,
            "predictions_generated": len(predictions),
            "run_date": str(run_date),
            "prediction_days": prediction_days,
            "message": f"Successfully generated and saved {len(predictions)} predictions for {prediction_days} days"
        }
        
    except Exception as e:
        logging.error(f"Error generating Prophet predictions: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "run_date": str(run_date) if 'run_date' in locals() else None
        }

@app.get("/api/prophet_predictions/top_gainers")
async def api_prophet_top_gainers(
    limit: int = Query(10, description="Number of top gainers to return"),
    prediction_days: int = Query(30, description="Number of prediction days to filter by (default: 30)"),
    force_refresh: bool = Query(False, description="Force refresh (ignore cache)")
):
    """API endpoint to get top N potential gainers based on Prophet predictions"""
    try:
        from stocks.ProphetPricePredictor import ProphetPricePredictor
        
        predictor = ProphetPricePredictor()
        top_gainers = predictor.get_top_gainers(limit=limit, prediction_days=prediction_days)
        
        # Validate that we have actual data
        if not top_gainers or len(top_gainers) == 0:
            # Check if predictions exist for this prediction_days
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM my_schema.prophet_predictions
                WHERE prediction_days = %s
                AND status = 'ACTIVE'
            """, (prediction_days,))
            
            result = cursor.fetchone()
            total_predictions = result[0] if result else 0
            cursor.close()
            conn.close()
            
            if total_predictions == 0:
                # No predictions exist for this prediction_days
                return {
                    "success": False,
                    "error": f"No predictions found for {prediction_days} days. Please generate predictions first.",
                    "top_gainers": [],
                    "count": 0,
                    "needs_generation": True
                }
            else:
                # Predictions exist but don't meet filter criteria (confidence, etc.)
                return {
                    "success": False,
                    "error": f"No top gainers found for {prediction_days} days matching the criteria (confidence >= 50%, positive gain). Try generating predictions with different parameters.",
                    "top_gainers": [],
                    "count": 0,
                    "needs_generation": False
                }
        
        # Calculate days_in_leaderboard for each stock
        if top_gainers and len(top_gainers) > 0:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get scrip_ids from top_gainers
            scrip_ids = [gainer.get('scrip_id') for gainer in top_gainers if gainer.get('scrip_id')]
            
            if scrip_ids:
                # Calculate days_in_leaderboard for all stocks in one query
                placeholders = ','.join(['%s'] * len(scrip_ids))
                cursor.execute(f"""
                    WITH ranked_predictions AS (
                        SELECT 
                            scrip_id,
                            run_date,
                            ROW_NUMBER() OVER (PARTITION BY run_date ORDER BY predicted_price_change_pct DESC) as rank
                        FROM my_schema.prophet_predictions
                        WHERE prediction_days = %s
                        AND status = 'ACTIVE'
                        AND predicted_price_change_pct > 0
                        AND prediction_confidence >= 50
                        AND scrip_id IN ({placeholders})
                    ),
                    top_10_dates AS (
                        SELECT DISTINCT scrip_id, run_date
                        FROM ranked_predictions
                        WHERE rank <= 10
                    )
                    SELECT scrip_id, COUNT(DISTINCT run_date) as days_count
                    FROM top_10_dates
                    GROUP BY scrip_id
                """, [prediction_days] + scrip_ids)
                
                days_counts = {row['scrip_id']: row['days_count'] for row in cursor.fetchall()}
                
                # Add days_in_leaderboard to each gainer
                for gainer in top_gainers:
                    scrip_id = gainer.get('scrip_id')
                    gainer['days_in_leaderboard'] = days_counts.get(scrip_id, 0)
            
            conn.close()
        
        return {
            "success": True,
            "top_gainers": top_gainers,
            "count": len(top_gainers)
        }
        
    except Exception as e:
        logging.error(f"Error getting top gainers: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e), "top_gainers": []}

@app.get("/api/prophet_predictions/days_in_leaderboard")
async def api_days_in_leaderboard(
    scrip_id: str = Query(..., description="Stock symbol"),
    prediction_days: int = Query(30, description="Prediction days to check (default: 30)")
):
    """API endpoint to count how many days a stock has been in Top 10 leaderboard"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get all distinct run_dates where this stock appeared in top 10
        # For each run_date, get top 10 stocks by predicted_price_change_pct
        cursor.execute("""
            WITH ranked_predictions AS (
                SELECT 
                    scrip_id,
                    run_date,
                    ROW_NUMBER() OVER (PARTITION BY run_date ORDER BY predicted_price_change_pct DESC) as rank
                FROM my_schema.prophet_predictions
                WHERE prediction_days = %s
                AND status = 'ACTIVE'
                AND predicted_price_change_pct > 0
                AND prediction_confidence >= 50
            ),
            top_10_dates AS (
                SELECT DISTINCT run_date
                FROM ranked_predictions
                WHERE scrip_id = %s
                AND rank <= 10
            )
            SELECT COUNT(DISTINCT run_date) as days_count
            FROM top_10_dates
        """, (prediction_days, scrip_id))
        
        result = cursor.fetchone()
        days_count = result['days_count'] if result else 0
        
        conn.close()
        
        return {
            "success": True,
            "scrip_id": scrip_id,
            "prediction_days": prediction_days,
            "days_in_leaderboard": days_count
        }
        
    except Exception as e:
        logging.error(f"Error counting days in leaderboard for {scrip_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e), "days_in_leaderboard": 0}

@app.get("/api/prophet_predictions/history/{scrip_id}")
async def api_prophet_prediction_history(
    scrip_id: str,
    prediction_days: int = Query(30, description="Prediction days to filter by (default: 30)"),
    limit: int = Query(100, ge=1, le=365, description="Maximum number of historical predictions to return")
):
    """API endpoint to get historical predictions for a stock to display in chart"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                run_date,
                current_price,
                predicted_price_change_pct,
                prediction_confidence,
                prediction_days
            FROM my_schema.prophet_predictions
            WHERE scrip_id = %s
            AND prediction_days = %s
            AND status = 'ACTIVE'
            ORDER BY run_date DESC
            LIMIT %s
        """, (scrip_id, prediction_days, limit))
        
        predictions = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts with proper date formatting
        history_list = []
        for pred in predictions:
            history_list.append({
                "run_date": str(pred['run_date']) if pred['run_date'] else None,
                "current_price": float(pred['current_price']) if pred['current_price'] else 0.0,
                "predicted_price_change_pct": float(pred['predicted_price_change_pct']) if pred['predicted_price_change_pct'] else 0.0,
                "prediction_confidence": float(pred['prediction_confidence']) if pred['prediction_confidence'] else 0.0,
                "prediction_days": int(pred['prediction_days']) if pred['prediction_days'] else prediction_days
            })
        
        # Reverse to show chronological order (oldest first)
        history_list.reverse()
        
        return {
            "success": True,
            "scrip_id": scrip_id,
            "prediction_days": prediction_days,
            "history": history_list
        }
        
    except Exception as e:
        logging.error(f"Error getting prediction history for {scrip_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e), "history": []}

@app.get("/api/prophet_predictions/symbol_search")
async def api_prophet_symbol_search(
    query: str = Query(..., description="Search query for stock symbol"),
    limit: int = Query(10, description="Maximum number of suggestions")
):
    """API endpoint to search for stock symbols with autocomplete"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Search in prophet_predictions for symbols that have predictions
        cursor.execute("""
            SELECT DISTINCT scrip_id
            FROM my_schema.prophet_predictions
            WHERE scrip_id ILIKE %s
            AND status = 'ACTIVE'
            ORDER BY scrip_id
            LIMIT %s
        """, (f"%{query}%", limit))
        
        suggestions = [row['scrip_id'] for row in cursor.fetchall()]
        
        # If not enough results, search in master_scrips table
        if len(suggestions) < limit:
            cursor.execute("""
                SELECT DISTINCT scrip_id
                FROM my_schema.master_scrips
                WHERE scrip_id ILIKE %s
                AND scrip_country = 'IN'
                AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                ORDER BY scrip_id
                LIMIT %s
            """, (f"%{query}%", limit - len(suggestions)))
            
            additional = [row['scrip_id'] for row in cursor.fetchall() if row['scrip_id'] not in suggestions]
            suggestions.extend(additional)
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "suggestions": suggestions[:limit]
        }
        
    except Exception as e:
        logging.error(f"Error searching symbols: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e), "suggestions": []}

@app.get("/api/prophet_predictions/{scrip_id}")
async def api_get_prophet_prediction(scrip_id: str, prediction_days: int = Query(60, description="Prediction days (default: 60)")):
    """API endpoint to get Prophet prediction for a specific stock"""
    try:
        from stocks.ProphetPricePredictor import ProphetPricePredictor
        
        predictor = ProphetPricePredictor()
        prediction = predictor.get_prediction_for_stock(scrip_id)
        
        if not prediction:
            return {
                "success": False,
                "error": f"No prediction found for {scrip_id}",
                "prediction": None
            }
        
        # Filter by prediction_days if specified
        if prediction.get('prediction_days') != prediction_days:
            # Try to get prediction for the specified days
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT * 
                FROM my_schema.prophet_predictions
                WHERE scrip_id = %s
                AND prediction_days = %s
                AND status = 'ACTIVE'
                ORDER BY run_date DESC
                LIMIT 1
            """, (scrip_id, prediction_days))
            
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if row:
                prediction = dict(row)
            else:
                return {
                    "success": False,
                    "error": f"No prediction found for {scrip_id} with {prediction_days} days",
                    "prediction": None
                }
        
        return {
            "success": True,
            "prediction": prediction
        }
        
    except Exception as e:
        logging.error(f"Error getting prediction for {scrip_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e), "prediction": None}

@app.get("/api/swing_trades_history")
async def api_swing_trades_history(
    start_date: str = Query(None, description="Start date YYYY-MM-DD"),
    end_date: str = Query(None, description="End date YYYY-MM-DD"),
    scrip_id: str = Query(None, description="Filter by stock symbol"),
    pattern_type: str = Query(None, description="Filter by pattern type"),
    limit: int = Query(100, ge=1, le=1000)
):
    """API endpoint to get historical swing trade recommendations"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        where_clauses = []
        params = []
        
        if start_date:
            where_clauses.append("analysis_date >= %s")
            params.append(start_date)
        if end_date:
            where_clauses.append("analysis_date <= %s")
            params.append(end_date)
        if scrip_id:
            where_clauses.append("scrip_id = %s")
            params.append(scrip_id)
        if pattern_type:
            where_clauses.append("pattern_type = %s")
            params.append(pattern_type)
        
        sql = f"""
            SELECT id, generated_at, analysis_date, run_date, scrip_id, instrument_token,
                   pattern_type, direction, entry_price, target_price, stop_loss,
                   potential_gain_pct, risk_reward_ratio, confidence_score, holding_period_days,
                   current_price, sma_20, sma_50, sma_200, rsi_14, macd, macd_signal, atr_14,
                   volume_trend, support_level, resistance_level, rationale,
                   technical_context, diagnostics, filtering_criteria, status
            FROM my_schema.swing_trade_suggestions
            {('WHERE ' + ' AND '.join(where_clauses)) if where_clauses else ''}
            ORDER BY analysis_date DESC, confidence_score DESC
            LIMIT %s
        """
        params.append(limit)
        
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        
        # Get Prophet predictions for all scrip_ids in the results
        scrip_ids = [row['scrip_id'] for row in rows if row.get('scrip_id')]
        predictions_map = {}
        
        if scrip_ids:
            try:
                cursor.execute("""
                    SELECT scrip_id, predicted_price_change_pct, prediction_confidence
                    FROM my_schema.prophet_predictions
                    WHERE scrip_id = ANY(%s)
                    AND run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE' AND prediction_days = 30)
                    AND prediction_days = 30
                    AND status = 'ACTIVE'
                """, (scrip_ids,))
                
                predictions_rows = cursor.fetchall()
                predictions_map = {row['scrip_id'].upper(): dict(row) for row in predictions_rows}
                logging.debug(f"Loaded {len(predictions_map)} Prophet predictions for history query")
            except Exception as e:
                logging.error(f"Error loading Prophet predictions for history: {e}")
        
        # Convert to list of dicts
        recommendations = []
        for row in rows:
            rec = dict(row)
            # Parse JSONB fields
            if rec.get('technical_context'):
                try:
                    if isinstance(rec['technical_context'], str):
                        rec['technical_context'] = json.loads(rec['technical_context'])
                except:
                    pass
            if rec.get('diagnostics'):
                try:
                    if isinstance(rec['diagnostics'], str):
                        rec['diagnostics'] = json.loads(rec['diagnostics'])
                except:
                    pass
            if rec.get('filtering_criteria'):
                try:
                    if isinstance(rec['filtering_criteria'], str):
                        rec['filtering_criteria'] = json.loads(rec['filtering_criteria'])
                except:
                    pass
            
            # Add Prophet predictions if available
            scrip_id = rec.get('scrip_id')
            if scrip_id and scrip_id.upper() in predictions_map:
                pred = predictions_map[scrip_id.upper()]
                rec['prophet_prediction_pct'] = float(pred['predicted_price_change_pct']) if pred['predicted_price_change_pct'] is not None else None
                rec['prophet_confidence'] = float(pred['prediction_confidence']) if pred['prediction_confidence'] is not None else None
            else:
                rec['prophet_prediction_pct'] = None
                rec['prophet_confidence'] = None
            
            recommendations.append(rec)
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "recommendations": recommendations,
            "total_found": len(recommendations)
        }
        
    except Exception as e:
        logging.error(f"Error fetching swing trade history: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.get("/api/scanner_with_confirmation")
async def api_scanner_with_confirmation(
    strategy_type: str = Query("covered_call", description="Strategy type"),
    expiry: str = Query(None, description="Expiry date"),
    start_time: str = Query("09:15:00", description="Start time for order flow analysis"),
    end_time: str = Query("15:30:00", description="End time for order flow analysis"),
    min_iv_rank: float = Query(50.0, description="Minimum IV Rank"),
    limit: int = Query(5, description="Number of candidates")
):
    """API endpoint to get options scanner results with order flow confirmation"""
    try:
        from options.OptionsScanner import OptionsScanner
        from market.MicroLevelDetector import MicroLevelDetector
        from market.FootprintChartGenerator import FootprintChartGenerator
        from datetime import datetime, date
        
        scanner = OptionsScanner()
        detector = MicroLevelDetector()
        footprint_gen = FootprintChartGenerator()
        
        # Parse expiry date
        expiry_date = None
        if expiry:
            try:
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            except:
                pass
        
        # Scan options
        candidates = scanner.scan_options_chain(
            expiry=expiry_date,
            strategy_type=strategy_type,
            min_iv_rank=min_iv_rank
        )
        
        # Limit results
        if limit > 0:
            candidates = candidates[:limit]
        
        # Get footprint data for confirmation
        analysis_date = date.today()
        footprint_data = footprint_gen.generate_footprint_data(
            start_time=start_time,
            end_time=end_time,
            analysis_date=analysis_date
        )
        
        # Get tactical levels aligned with scanner
        if candidates and footprint_data.get('success'):
            tactical_data = detector.get_tactical_levels_for_scanner(
                candidates, footprint_data
            )
        else:
            tactical_data = {'tactical_levels': [], 'aligned_candidates': [], 'confirmation_rate': 0}
        
        return {
            'success': True,
            'scanner_candidates': candidates[:limit],
            'footprint_data': footprint_data,
            'tactical_levels': tactical_data.get('tactical_levels', []),
            'aligned_candidates': tactical_data.get('aligned_candidates', []),
            'confirmation_rate': tactical_data.get('confirmation_rate', 0)
        }
    except Exception as e:
        logging.error(f"Error generating scanner with confirmation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.get("/api/gainers")
async def api_gainers():
    """API endpoint to get top 10 gainers from last trading day"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        logging.info("Starting gainers query...")
        
        # First check if we have data
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM my_schema.rt_intraday_price 
            WHERE country = 'IN' 
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """)
        total_rows = cursor.fetchone()['cnt']
        logging.info(f"Total rows in rt_intraday_price: {total_rows}")
        
        # Get latest date
        cursor.execute("""
            SELECT MAX(price_date::date) as max_date
            FROM my_schema.rt_intraday_price
            WHERE country = 'IN'
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """)
        latest_result = cursor.fetchone()
        latest_date = latest_result['max_date'] if latest_result else None
        logging.info(f"Latest date: {latest_date}")
        
        if not latest_date:
            logging.warning("No latest date found, returning empty gainers")
            conn.close()
            return cached_json_response({"gainers": []}, "/api/gainers")
        
        # Get previous date
        cursor.execute("""
            SELECT MAX(price_date::date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE country = 'IN'
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
            AND price_date::date < %s
        """, (latest_date,))
        prev_result = cursor.fetchone()
        prev_date = prev_result['prev_date'] if prev_result else None
        logging.info(f"Previous date: {prev_date}")
        
        if not prev_date:
            logging.warning("No previous date found, returning empty gainers")
            conn.close()
            return cached_json_response({"gainers": []}, "/api/gainers")
        
        # Simplified query using CTEs for better readability and debugging
        # Use the calculated latest_date and prev_date to ensure correct date comparison
        cursor.execute("""
            select "Curr".scrip_id, 
                   100*("Curr".price_close - "Prev".price_close)/"Prev".price_close "Gain",
                   "Curr".price_close as current_price,
                   "Prev".price_close as previous_price
            from my_schema.master_scrips ms,
            (
                select SCRIP_ID, PRICE_CLOSE from my_schema.rt_intraday_price rip 
                where price_date::date = %s
                and country = 'IN'
                and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
            ) "Curr",
            (
                select SCRIP_ID, PRICE_CLOSE from my_schema.rt_intraday_price rip 
                where price_date::date = %s
                and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                and country = 'IN'
            ) "Prev"
            where "Curr".scrip_id = "Prev".scrip_id
            and "Curr".scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH') 
            and "Curr".scrip_id = ms.scrip_id
            and ms.scrip_country = 'IN'
            order by 100*("Curr".price_close - "Prev".price_close)/"Prev".price_close desc
            limit 10
        """, (latest_date, prev_date))
        
        gainers = cursor.fetchall()
        logging.info(f"Query returned {len(gainers)} rows from database")
        
        # Debug: Check counts at each step to diagnose why we might get 0 results
        if len(gainers) == 0:
            logging.warning("Main query returned 0 results, checking data availability...")
            # Check how many rows match current date
            cursor.execute("""
                SELECT COUNT(*) as cnt
                FROM my_schema.rt_intraday_price
                WHERE price_date::date = %s
                AND country = 'IN'
                AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                AND price_close IS NOT NULL
                AND price_close > 0
            """, (latest_date,))
            current_date_count = cursor.fetchone()['cnt']
            logging.info(f"Rows for latest date {latest_date}: {current_date_count}")
            
            # Check how many rows match previous date
            cursor.execute("""
                SELECT COUNT(*) as cnt
                FROM my_schema.rt_intraday_price
                WHERE price_date::date = %s
                AND country = 'IN'
                AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                AND price_close IS NOT NULL
                AND price_close > 0
            """, (prev_date,))
            prev_date_count = cursor.fetchone()['cnt']
            logging.info(f"Rows for previous date {prev_date}: {prev_date_count}")
            
            # Check how many scrip_ids match between dates
            cursor.execute("""
                SELECT COUNT(DISTINCT cp.scrip_id) as match_count
                FROM (
                    SELECT DISTINCT scrip_id
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                    AND country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                    AND price_close IS NOT NULL
                    AND price_close > 0
                ) cp
                INNER JOIN (
                    SELECT DISTINCT scrip_id
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                    AND country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                    AND price_close IS NOT NULL
                    AND price_close > 0
                ) pp ON cp.scrip_id = pp.scrip_id
            """, (latest_date, prev_date))
            match_count = cursor.fetchone()['match_count']
            logging.info(f"Matching scrip_ids between dates: {match_count}")
        
        logging.info(f"Final: Query returned {len(gainers)} rows from database")
        
        gainers_list = []
        for row in gainers:
            try:
                logging.debug(f"Processing gainer row: {row}")
                # Handle case-sensitive column names from SQL query
                gain_value = row.get('gain') or row.get('Gain') or 0.0
                current_price_value = row.get('current_price') or row.get('CURRENT_PRICE') or 0.0
                previous_price_value = row.get('previous_price') or row.get('PREVIOUS_PRICE') or 0.0
                
                gainers_list.append({
                    "scrip_id": row.get('scrip_id') or row.get('SCRIP_ID') or row.get('scrip_id'),
                    "gain": float(gain_value) if gain_value is not None else 0.0,
                    "current_price": float(current_price_value) if current_price_value is not None else 0.0,
                    "previous_price": float(previous_price_value) if previous_price_value is not None else 0.0
                })
            except (ValueError, TypeError, KeyError) as e:
                logging.warning(f"Error processing gainer row for {row.get('scrip_id') or row.get('SCRIP_ID', 'unknown')}: {e}")
                logging.warning(f"Row data: {dict(row)}")
                logging.warning(f"Available keys: {list(row.keys()) if hasattr(row, 'keys') else 'N/A'}")
                continue
        
        conn.close()
        logging.info(f"Fetched {len(gainers_list)} gainers, returning response")
        logging.debug(f"Gainers list: {gainers_list}")
        return cached_json_response({
            "gainers": gainers_list,
            "latest_date": str(latest_date) if latest_date else None,
            "previous_date": str(prev_date) if prev_date else None
        }, "/api/gainers")
    except Exception as e:
        logging.error(f"Error fetching gainers: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"error": str(e), "gainers": []}, "/api/gainers")

@app.get("/api/losers")
async def api_losers():
    """API endpoint to get top 10 losers from last trading day"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        logging.info("Starting losers query...")
        
        # First check if we have data
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM my_schema.rt_intraday_price 
            WHERE country = 'IN' 
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """)
        total_rows = cursor.fetchone()['cnt']
        logging.info(f"Total rows in rt_intraday_price: {total_rows}")
        
        # Get latest date
        cursor.execute("""
            SELECT MAX(price_date::date) as max_date
            FROM my_schema.rt_intraday_price
            WHERE country = 'IN'
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """)
        latest_result = cursor.fetchone()
        latest_date = latest_result['max_date'] if latest_result else None
        logging.info(f"Latest date: {latest_date}")
        
        if not latest_date:
            logging.warning("No latest date found, returning empty losers")
            conn.close()
            return cached_json_response({"losers": []}, "/api/losers")
        
        # Get previous date
        cursor.execute("""
            SELECT MAX(price_date::date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE country = 'IN'
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
            AND price_date::date < %s
        """, (latest_date,))
        prev_result = cursor.fetchone()
        prev_date = prev_result['prev_date'] if prev_result else None
        logging.info(f"Previous date: {prev_date}")
        
        if not prev_date:
            logging.warning("No previous date found, returning empty losers")
            conn.close()
            return cached_json_response({"losers": []}, "/api/losers")
        
        # Simplified query using CTEs for better readability and debugging
        # Use the calculated latest_date and prev_date to ensure correct date comparison
        cursor.execute("""
            select "Curr".scrip_id, 
                   100*("Curr".price_close - "Prev".price_close)/"Prev".price_close "Gain",
                   "Curr".price_close as current_price,
                   "Prev".price_close as previous_price
            from my_schema.master_scrips ms,
            (
                select SCRIP_ID, PRICE_CLOSE from my_schema.rt_intraday_price rip 
                where price_date::date = %s
                and country = 'IN'
                and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
            ) "Curr",
            (
                select SCRIP_ID, PRICE_CLOSE from my_schema.rt_intraday_price rip 
                where price_date::date = %s
                and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                and country = 'IN'
            ) "Prev"
            where "Curr".scrip_id = "Prev".scrip_id
            and "Curr".scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH') 
            and "Curr".scrip_id = ms.scrip_id
            and ms.scrip_country = 'IN'
            order by 100*("Prev".price_close - "Curr".price_close)/"Prev".price_close desc
            limit 10
        """, (latest_date, prev_date))
        
        losers = cursor.fetchall()
        logging.info(f"Query returned {len(losers)} rows from database")
        
        # Debug: Check counts at each step to diagnose why we might get 0 results
        if len(losers) == 0:
            logging.warning("Main query returned 0 results, checking data availability...")
            # Check how many rows match current date
            cursor.execute("""
                SELECT COUNT(*) as cnt
                FROM my_schema.rt_intraday_price
                WHERE price_date::date = %s
                AND country = 'IN'
                AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                AND price_close IS NOT NULL
                AND price_close > 0
            """, (latest_date,))
            current_date_count = cursor.fetchone()['cnt']
            logging.info(f"Rows for latest date {latest_date}: {current_date_count}")
            
            # Check how many rows match previous date
            cursor.execute("""
                SELECT COUNT(*) as cnt
                FROM my_schema.rt_intraday_price
                WHERE price_date::date = %s
                AND country = 'IN'
                AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                AND price_close IS NOT NULL
                AND price_close > 0
            """, (prev_date,))
            prev_date_count = cursor.fetchone()['cnt']
            logging.info(f"Rows for previous date {prev_date}: {prev_date_count}")
            
            # Check how many scrip_ids match between dates
            cursor.execute("""
                SELECT COUNT(DISTINCT cp.scrip_id) as match_count
                FROM (
                    SELECT DISTINCT scrip_id
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                    AND country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                    AND price_close IS NOT NULL
                    AND price_close > 0
                ) cp
                INNER JOIN (
                    SELECT DISTINCT scrip_id
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                    AND country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                    AND price_close IS NOT NULL
                    AND price_close > 0
                ) pp ON cp.scrip_id = pp.scrip_id
            """, (latest_date, prev_date))
            match_count = cursor.fetchone()['match_count']
            logging.info(f"Matching scrip_ids between dates: {match_count}")
        
        logging.info(f"Final: Query returned {len(losers)} rows from database")
        
        losers_list = []
        for row in losers:
            try:
                logging.debug(f"Processing loser row: {row}")
                losers_list.append({
                    "scrip_id": row['scrip_id'],
                    "gain": float(row['Gain']) if row['Gain'] is not None else 0.0,
                    "current_price": float(row['current_price']) if row['current_price'] is not None else 0.0,
                    "previous_price": float(row['previous_price']) if row['previous_price'] is not None else 0.0
                })
            except (ValueError, TypeError) as e:
                logging.warning(f"Error processing loser row for {row.get('scrip_id', 'unknown')}: {e}")
                logging.warning(f"Row data: {dict(row)}")
                continue
        
        conn.close()
        logging.info(f"Fetched {len(losers_list)} losers, returning response")
        logging.debug(f"Losers list: {losers_list}")
        return cached_json_response({
            "losers": losers_list,
            "latest_date": str(latest_date) if latest_date else None,
            "previous_date": str(prev_date) if prev_date else None
        }, "/api/losers")
    except Exception as e:
        logging.error(f"Error fetching losers: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"error": str(e), "losers": []}, "/api/losers")

@app.get("/api/rt_intraday_price/latest")
async def api_rt_intraday_price_latest():
    """API endpoint to check for latest updates in rt_intraday_price table"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get latest date and timestamp
        cursor.execute("""
            SELECT 
                MAX(price_date::date) as latest_date,
                MAX(created_at) as latest_timestamp
            FROM my_schema.rt_intraday_price
            WHERE country = 'IN'
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        latest_date = result['latest_date'] if result else None
        latest_timestamp = result['latest_timestamp'] if result else None
        
        # Return without cache headers for real-time polling
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "success": True,
            "latest_date": str(latest_date) if latest_date else None,
            "latest_timestamp": str(latest_timestamp) if latest_timestamp else None
        }, headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        })
        
    except Exception as e:
        logging.error(f"Error checking latest rt_intraday_price: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.get("/api/holdings")
async def api_holdings(page: int = Query(1, ge=1), per_page: int = Query(10, ge=1), sort_by: str = Query(None), sort_dir: str = Query("asc"), search: str = Query(None)):
    """API endpoint to get paginated holdings data with sorting and GTT info"""
    try:
        # Small cache for hot endpoint (exclude search from cache key for real-time search)
        if not search:
            cache_key = f"holdings:{page}:{per_page}:{sort_by}:{sort_dir}"
            cached = cache_get_json(cache_key)
            if cached:
                return cached
        # If sorting by today_pnl or prophet_prediction_pct, we need to get all holdings, enrich them, sort, then paginate
        if sort_by == 'today_pnl' or sort_by == 'prophet_prediction_pct':
            # Get all holdings without pagination
            holdings_info = get_holdings_data(page=1, per_page=10000, sort_by='trading_symbol', sort_dir='asc', search=search)
        else:
            holdings_info = get_holdings_data(page=page, per_page=per_page, sort_by=sort_by, sort_dir=sort_dir, search=search)
        
        # Get all active GTTs for holdings lookup
        from kite.KiteGTT import KiteGTTManager
        manager = KiteGTTManager(kite)
        all_gtts = manager.get_all_gtts()
        
        # Create a map of tradingsymbol to GTT info
        gtt_map = {}
        for gtt in all_gtts:
            symbol = gtt.get('tradingsymbol', '')
            if symbol:
                gtt_map[symbol] = {
                    'trigger_id': gtt.get('trigger_id') or gtt.get('id'),
                    'trigger_price': gtt.get('trigger_price'),
                    'order_price': gtt.get('order_price'),
                    'status': gtt.get('status', 'ACTIVE')
                }
        
        # Get today's P&L prices for all holdings
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get today's and previous day's prices
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
            and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        # Create a map of today's P&L per holding by instrument_token
        today_pnl_map = {}
        
        # Always populate the map with holdings first, even if prev_date doesn't exist (will default to 0)
        # First, get all holdings to ensure we have entries for all of them
        cursor.execute("""
            SELECT instrument_token, trading_symbol, quantity
            FROM my_schema.holdings
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
        """)
        
        for row in cursor.fetchall():
            instrument_token = row['instrument_token']
            # Initialize with 0 values for all holdings
            today_pnl_map[instrument_token] = {
                'today_pnl': 0.0,
                'today_price': 0.0,
                'prev_price': 0.0,
                'pct_change': 0.0,
                'today_change': '0 (0%)'
            }
        
        if prev_date:
            # Convert prev_date to string format safely
            try:
                if hasattr(prev_date, 'strftime'):
                    prev_date_str = prev_date.strftime('%Y-%m-%d')
                elif isinstance(prev_date, str):
                    prev_date_str = prev_date
                else:
                    prev_date_str = str(prev_date)
            except Exception as e:
                logging.error(f"Error converting prev_date to string: {e}, prev_date={prev_date}")
                prev_date_str = None
            
            if prev_date_str:
                cursor.execute("""
                    WITH holdings_today AS (
                        SELECT instrument_token, trading_symbol, quantity, last_price
                        FROM my_schema.holdings
                        WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                    ),
                    today_prices AS (
                        SELECT scrip_id, price_close
                        FROM my_schema.rt_intraday_price
                        WHERE price_date = %s
                    ),
                    latest_prices AS (
                        SELECT scrip_id, price_close, 
                               ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                        FROM my_schema.rt_intraday_price
                        WHERE price_date::date <= CURRENT_DATE
                    ),
                    prev_prices AS (
                        SELECT scrip_id, price_close
                        FROM my_schema.rt_intraday_price
                        WHERE price_date = %s
                    )
                    SELECT 
                        h.instrument_token,
                        h.trading_symbol,
                        h.quantity,
                        COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                        COALESCE(prev_p.price_close, 0) as prev_price,
                        -- If today's price is null/0, P&L should be 0 regardless of prev price
                        -- If prev_price is 0, P&L should be 0 (no previous data to compare)
                        CASE 
                            WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                            WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                            ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                        END as today_pnl,
                        round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
    	                    ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
    	                else 0
    	                end::numeric, 2) as pct_change,
                        concat(round(CASE 
                            WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                            WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                            ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                        END), ' (',
                        round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
    	                    ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
    	                else 0
    	                end::numeric, 2), '%%)') as "Todays_Change"
                    FROM holdings_today h
                    LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                    LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                    LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
                """, (today_str, prev_date_str))
                
                for row in cursor.fetchall():
                    instrument_token = row['instrument_token']
                    today_price = float(row['today_price']) if row['today_price'] else 0.0
                    prev_price = float(row['prev_price']) if row['prev_price'] else 0.0
                    today_pnl = float(row['today_pnl']) if row['today_pnl'] else 0.0
                    pct_change = float(row['pct_change']) if row['pct_change'] else 0.0
                    today_change_str = str(row.get("Todays_Change", '')) if row.get("Todays_Change") else ''
                    
                    # Store by instrument_token to match individual holdings
                    today_pnl_map[instrument_token] = {
                        'today_pnl': today_pnl,
                        'today_price': today_price,
                        'prev_price': prev_price,
                        'pct_change': pct_change,
                        'today_change': today_change_str
                    }
        
        # Get Prophet predictions for all holdings
        conn2 = get_db_connection()
        cursor2 = conn2.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        try:
            # Get Prophet predictions - prefer 60-day, fallback to latest available
            cursor2.execute("""
                SELECT DISTINCT ON (scrip_id) 
                    scrip_id, 
                    predicted_price_change_pct, 
                    prediction_confidence,
                    prediction_days,
                    prediction_details
                FROM my_schema.prophet_predictions
                WHERE status = 'ACTIVE'
                ORDER BY scrip_id, 
                         CASE WHEN prediction_days = 60 THEN 1 
                              WHEN prediction_days = 30 THEN 2 
                              ELSE 3 END,
                         run_date DESC
            """)
            
            predictions_rows = cursor2.fetchall()
            predictions_map = {row['scrip_id'].upper(): dict(row) for row in predictions_rows}
            logging.debug(f"Loaded {len(predictions_map)} Prophet predictions for holdings")
        except Exception as e:
            logging.error(f"Error loading Prophet predictions for holdings: {e}")
            predictions_map = {}
        finally:
            cursor2.close()
            conn2.close()
        
        conn.close()
        
        # Convert holdings to serializable format
        holdings_list = []
        for holding in holdings_info["holdings"]:
            symbol = holding["trading_symbol"]
            instrument_token = holding["instrument_token"]
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'today_price': 0.0, 'prev_price': 0.0, 'pct_change': 0.0, 'today_change': '0 (0%)'})
            
            # Get Prophet prediction for this holding
            prophet_pred = None
            prophet_conf = None
            prophet_pred_days = None
            prophet_cv_mape = None
            prophet_cv_rmse = None
            if symbol and symbol.upper() in predictions_map:
                pred = predictions_map[symbol.upper()]
                try:
                    prophet_pred = float(pred['predicted_price_change_pct']) if pred.get('predicted_price_change_pct') is not None else None
                    prophet_conf = float(pred['prediction_confidence']) if pred.get('prediction_confidence') is not None else None
                    prophet_pred_days = int(pred['prediction_days']) if pred.get('prediction_days') is not None else None
                    
                    # Extract cv_metrics from prediction_details
                    prediction_details = pred.get('prediction_details')
                    if prediction_details:
                        try:
                            # PostgreSQL JSONB might return as dict or string depending on psycopg2 version
                            if isinstance(prediction_details, str):
                                details = json.loads(prediction_details)
                            elif isinstance(prediction_details, dict):
                                details = prediction_details
                            else:
                                # Try to convert to string first
                                details = json.loads(str(prediction_details))
                            
                            cv_metrics = details.get('cv_metrics')
                            if cv_metrics and isinstance(cv_metrics, dict):
                                # Handle None, infinity, or invalid values
                                mape_value = cv_metrics.get('mape')
                                rmse_value = cv_metrics.get('rmse')
                                
                                # Convert infinity strings or infinity values to None
                                if mape_value is None or mape_value == float('inf') or mape_value == float('-inf') or (isinstance(mape_value, str) and mape_value.lower() in ['inf', 'infinity', 'nan']):
                                    prophet_cv_mape = None
                                else:
                                    try:
                                        prophet_cv_mape = float(mape_value) if mape_value is not None else None
                                    except (ValueError, TypeError):
                                        prophet_cv_mape = None
                                
                                if rmse_value is None or rmse_value == float('inf') or rmse_value == float('-inf') or (isinstance(rmse_value, str) and rmse_value.lower() in ['inf', 'infinity', 'nan']):
                                    prophet_cv_rmse = None
                                else:
                                    try:
                                        prophet_cv_rmse = float(rmse_value) if rmse_value is not None else None
                                    except (ValueError, TypeError):
                                        prophet_cv_rmse = None
                                
                                logging.debug(f"Extracted cv_metrics for {symbol}: MAPE={prophet_cv_mape}, RMSE={prophet_cv_rmse}")
                            else:
                                prophet_cv_mape = None
                                prophet_cv_rmse = None
                                logging.debug(f"No cv_metrics found in prediction_details for {symbol}, cv_metrics={cv_metrics}")
                        except (json.JSONDecodeError, Exception) as e:
                            logging.warning(f"Error parsing prediction_details for {symbol}: {e}, type={type(prediction_details)}")
                except (ValueError, TypeError):
                    prophet_pred = None
                    prophet_conf = None
                    prophet_pred_days = None
                    prophet_cv_mape = None
                    prophet_cv_rmse = None
            
            holdings_list.append({
                "trading_symbol": symbol,
                "instrument_token": instrument_token,
                "quantity": holding["quantity"],
                "average_price": float(holding["average_price"]) if holding["average_price"] else 0.0,
                "last_price": float(holding["last_price"]) if holding["last_price"] else 0.0,
                "pnl": float(holding["pnl"]) if holding["pnl"] else 0.0,
                "invested_amount": float(holding.get("invested_amount", 0)),
                "current_amount": float(holding.get("current_amount", 0)),
                "pnl_pct_change": float(holding.get("pnl_pct_change", 0)) if holding.get("pnl_pct_change") else 0.0,
                "today_pnl": today_pnl_info['today_pnl'],
                "today_price": today_pnl_info['today_price'],
                "prev_price": today_pnl_info['prev_price'],
                "pct_change": today_pnl_info.get('pct_change', 0.0),
                "today_change": today_pnl_info.get('today_change', '0 (0%)'),
                "prophet_prediction_pct": prophet_pred,
                "prophet_confidence": prophet_conf,
                "prediction_days": prophet_pred_days,
                "prophet_cv_mape": prophet_cv_mape,
                "prophet_cv_rmse": prophet_cv_rmse
            })
        
        # If sorting by today_pnl or prophet_prediction_pct, sort the enriched list and then paginate
        if sort_by == 'today_pnl':
            sort_reverse = sort_dir.lower() == 'desc'
            holdings_list.sort(key=lambda x: x['today_pnl'], reverse=sort_reverse)
            
            # Apply pagination after sorting
            total_count = len(holdings_list)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            holdings_list = holdings_list[start_idx:end_idx]
        elif sort_by == 'prophet_prediction_pct':
            sort_reverse = sort_dir.lower() == 'desc'
            # Sort by prophet_prediction_pct, handling None values
            holdings_list.sort(key=lambda x: x['prophet_prediction_pct'] if x['prophet_prediction_pct'] is not None else (float('-inf') if sort_reverse else float('inf')), reverse=sort_reverse)
            
            # Apply pagination after sorting
            total_count = len(holdings_list)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            holdings_list = holdings_list[start_idx:end_idx]
        else:
            total_count = holdings_info["total_count"]
        
        result = {
            "holdings": holdings_list,
            "total_count": total_count,
            "page": page,
            "per_page": per_page
        }
        cache_set_json(cache_key, result, ttl_seconds=5)
        return cached_json_response(result, "/api/holdings")
    except Exception as e:
        logging.error(f"Error fetching holdings: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"error": str(e)}, "/api/holdings")

@app.get("/api/positions")
async def api_positions():
    """API endpoint to get latest positions data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                fetch_timestamp, 
                position_type, 
                trading_symbol, 
                product, 
                exchange, 
                average_price, 
                pnl 
            FROM my_schema.positions 
            WHERE run_date = CURRENT_DATE
            ORDER BY trading_symbol
        """)
        
        positions = cursor.fetchall()
        conn.close()
        
        # Convert to serializable format
        positions_list = []
        for position in positions:
            # Handle fetch_timestamp (could be string or datetime)
            fetch_ts = position["fetch_timestamp"]
            if fetch_ts:
                if hasattr(fetch_ts, 'strftime'):
                    fetch_ts_str = fetch_ts.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    fetch_ts_str = str(fetch_ts)
            else:
                fetch_ts_str = ""
            
            positions_list.append({
                "fetch_timestamp": fetch_ts_str,
                "position_type": position["position_type"],
                "trading_symbol": position["trading_symbol"],
                "product": position["product"],
                "exchange": position["exchange"],
                "average_price": float(position["average_price"]) if position["average_price"] else 0.0,
                "pnl": float(position["pnl"]) if position["pnl"] else 0.0
            })
        
        return cached_json_response({"positions": positions_list}, "/api/positions")
    except Exception as e:
        logging.error(f"Error fetching positions: {e}")
        return cached_json_response({"error": str(e), "positions": []}, "/api/positions")

@app.get("/api/mf_holdings")
async def api_mf_holdings():
    """API endpoint to get latest MF holdings data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                folio,
                fund,
                tradingsymbol,
                isin,
                quantity,
                average_price,
                last_price,
                invested_amount,
                current_value,
                pnl,
                net_change_percentage,
                day_change_percentage
            FROM my_schema.mf_holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
            ORDER BY fund, tradingsymbol
        """)
        
        mf_holdings = cursor.fetchall()
        conn.close()
        
        # Convert to serializable format
        mf_holdings_list = []
        for mf in mf_holdings:
            mf_holdings_list.append({
                "folio": mf["folio"],
                "fund": mf["fund"],
                "tradingsymbol": mf["tradingsymbol"],
                "isin": mf["isin"],
                "quantity": float(mf["quantity"]) if mf["quantity"] else 0.0,
                "average_price": float(mf["average_price"]) if mf["average_price"] else 0.0,
                "last_price": float(mf["last_price"]) if mf["last_price"] else 0.0,
                "invested_amount": float(mf["invested_amount"]) if mf["invested_amount"] else 0.0,
                "current_value": float(mf["current_value"]) if mf["current_value"] else 0.0,
                "pnl": float(mf["pnl"]) if mf["pnl"] else 0.0,
                "net_change_percentage": float(mf["net_change_percentage"]) if mf["net_change_percentage"] else 0.0,
                "day_change_percentage": float(mf["day_change_percentage"]) if mf["day_change_percentage"] else 0.0
            })
        
        return cached_json_response({"mf_holdings": mf_holdings_list}, "/api/mf_holdings")
        
    except Exception as e:
        logging.error(f"Error fetching MF holdings: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"error": str(e), "mf_holdings": []}, "/api/mf_holdings")

@app.get("/api/mf_nav/{mf_symbol}")
async def api_mf_nav(
    mf_symbol: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(365, description="Maximum number of records")
):
    """API endpoint to get historical NAV data for a mutual fund"""
    try:
        from datetime import datetime
        from holdings.MFNAVFetcher import MFNAVFetcher
        
        # Parse dates
        start = None
        end = None
        
        if start_date:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT nav_date, nav_value, scheme_code, fund_name
            FROM my_schema.mf_nav_history
            WHERE mf_symbol = %s
        """
        params = [mf_symbol]
        
        if start:
            query += " AND nav_date >= %s"
            params.append(start)
        if end:
            query += " AND nav_date <= %s"
            params.append(end)
        
        query += " ORDER BY nav_date DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        nav_records = cursor.fetchall()
        cursor.close()
        conn.close()
        
        nav_list = []
        for record in nav_records:
            nav_list.append({
                "nav_date": record["nav_date"].strftime('%Y-%m-%d') if record["nav_date"] else None,
                "nav_value": float(record["nav_value"]) if record["nav_value"] else 0.0,
                "scheme_code": record["scheme_code"],
                "fund_name": record["fund_name"]
            })
        
        return cached_json_response({
            "success": True,
            "mf_symbol": mf_symbol,
            "nav_data": nav_list,
            "count": len(nav_list)
        }, f"/api/mf_nav/{mf_symbol}")
        
    except Exception as e:
        logging.error(f"Error fetching NAV data for {mf_symbol}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({
            "success": False,
            "error": str(e),
            "mf_symbol": mf_symbol,
            "nav_data": []
        }, f"/api/mf_nav/{mf_symbol}")

@app.get("/api/mf_benchmark_comparison")
async def api_mf_benchmark_comparison(
    mf_symbol: Optional[str] = Query(None, description="Mutual Fund symbol (if not provided, returns for all MFs)"),
    benchmark_symbol: Optional[str] = Query(None, description="Benchmark symbol (auto-detected if not provided)")
):
    """API endpoint to get MF performance vs benchmark metrics"""
    try:
        from holdings.MFPerformanceAnalyzer import MFPerformanceAnalyzer
        
        analyzer = MFPerformanceAnalyzer()
        
        if mf_symbol:
            # Get benchmark for this specific MF
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT tradingsymbol, fund
                FROM my_schema.mf_holdings
                WHERE tradingsymbol = %s
                LIMIT 1
            """, (mf_symbol,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not result:
                return cached_json_response({
                    "success": False,
                    "error": f"MF {mf_symbol} not found in holdings"
                }, "/api/mf_benchmark_comparison")
            
            fund_name = result[1]
            performance = analyzer.analyze_mf_performance(mf_symbol, fund_name, benchmark_symbol)
            
            return cached_json_response(performance, "/api/mf_benchmark_comparison")
        else:
            # Get comparison for all MFs
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT tradingsymbol, fund
                FROM my_schema.mf_holdings
                WHERE tradingsymbol IS NOT NULL AND tradingsymbol != ''
                ORDER BY tradingsymbol
            """)
            
            mf_list = cursor.fetchall()
            cursor.close()
            conn.close()
            
            comparisons = []
            for mf_symbol_item, fund_name in mf_list:
                try:
                    performance = analyzer.analyze_mf_performance(mf_symbol_item, fund_name)
                    if performance.get('success'):
                        comparisons.append(performance)
                except Exception as e:
                    logging.warning(f"Error analyzing {mf_symbol_item}: {e}")
                    continue
            
            return cached_json_response({
                "success": True,
                "comparisons": comparisons,
                "count": len(comparisons)
            }, "/api/mf_benchmark_comparison")
        
    except Exception as e:
        logging.error(f"Error fetching benchmark comparison: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({
            "success": False,
            "error": str(e)
        }, "/api/mf_benchmark_comparison")

@app.get("/api/holdings/patterns")
async def api_holdings_patterns():
    """API endpoint to get detected patterns for all holdings"""
    try:
        from stocks.SwingTradeScanner import SwingTradeScanner
        scanner = SwingTradeScanner(min_gain=10.0, max_gain=20.0, min_confidence=70.0)
        
        # Get holdings symbols
        holdings_info = get_holdings_data(page=1, per_page=10000, sort_by='trading_symbol', sort_dir='asc')
        symbols = [h["trading_symbol"] for h in holdings_info.get("holdings", [])]
        
        # Get patterns for each holding
        patterns_map = {}
        for symbol in symbols:
            try:
                patterns = scanner.get_patterns_for_stock(symbol)
                if patterns:
                    patterns_map[symbol] = patterns
            except Exception as e:
                logging.debug(f"Error getting patterns for {symbol}: {e}")
                continue
        
        return cached_json_response({"success": True, "patterns": patterns_map}, "/api/holdings/patterns")
        
    except Exception as e:
        logging.error(f"Error fetching holdings patterns: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"success": False, "error": str(e), "patterns": {}}, "/api/holdings/patterns")

@app.get("/api/today_pnl_summary")
async def api_today_pnl_summary():
    """API endpoint to get today's P&L summary for Equity, MF, and Intraday trades"""
    try:
        cached = cache_get_json("today_pnl_summary")
        if cached:
            return cached
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Find the latest date that has actual stock data (exclude crypto symbols)
        # This prevents future-dated crypto data from skewing the results
        cursor.execute("""
            SELECT MAX(price_date::date) as latest_date
            FROM my_schema.rt_intraday_price
            WHERE country = 'IN'
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """)
        latest_result = cursor.fetchone()
        latest_date = latest_result['latest_date'] if latest_result and latest_result['latest_date'] else None
        
        if not latest_date:
            logging.warning("No stock data found in rt_intraday_price (excluding crypto), using CURRENT_DATE")
            cursor.execute("SELECT CURRENT_DATE as today_date")
            today = cursor.fetchone()['today_date']
            latest_date = today
        
        latest_date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
        logging.info(f"Using latest stock data date: {latest_date_str} (excluding crypto symbols)")
        
        # Get previous trading day (also excluding crypto)
        cursor.execute("""
            SELECT MAX(price_date::date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date::date < %s
            AND country = 'IN'
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """, (latest_date,))
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        # Calculate Equity P&L - sum up individual holdings P&L
        equity_pnl = 0.0
        if prev_date:
            # Convert prev_date to string format
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT instrument_token, trading_symbol, quantity, last_price
                    FROM my_schema.holdings
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                    AND country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                ),
                latest_prices AS (
                    SELECT scrip_id, price_close, 
                           ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= %s
                    AND country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                    AND country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                )
                SELECT 
                    h.instrument_token,
                    h.trading_symbol,
                    h.quantity,
                    COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    -- If either today's or previous price is missing/0, P&L should be 0
                    CASE 
                        WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                        WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                        ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) - prev_p.price_close)
                    END as today_pnl
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (latest_date_str, latest_date_str, prev_date_str))
            
            # Sum up all individual holdings P&L
            rows = cursor.fetchall()
            logging.info(f"Found {len(rows)} holdings for P&L calculation")
            for row in rows:
                holding_pnl = float(row['today_pnl']) if row['today_pnl'] else 0.0
                equity_pnl += holding_pnl
                logging.debug(f"P&L for {row.get('trading_symbol', 'unknown')}: {holding_pnl}")
            
            logging.info(f"Total equity P&L: {equity_pnl}")
        
        # Calculate MF P&L (today's change)
        cursor.execute("""
            SELECT 
                COALESCE(SUM(day_change_percentage * invested_amount / 100), 0) as mf_pnl
            FROM my_schema.mf_holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
        """)
        mf_result = cursor.fetchone()
        mf_pnl = float(mf_result['mf_pnl']) if mf_result and mf_result['mf_pnl'] else 0.0
        
        # Calculate Intraday trades P&L (from positions)
        cursor.execute("""
            SELECT 
                COALESCE(SUM(pnl), 0) as intraday_pnl
            FROM my_schema.positions 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.positions)
            AND position_type = 'day'
        """)
        intraday_result = cursor.fetchone()
        intraday_pnl = float(intraday_result['intraday_pnl']) if intraday_result and intraday_result['intraday_pnl'] else 0.0
        
        # Calculate overall Equity P&L (total unrealized P&L from holdings)
        cursor.execute("""
            SELECT 
                COALESCE(SUM(pnl), 0) as overall_equity_pnl
            FROM my_schema.holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
        """)
        equity_overall_result = cursor.fetchone()
        equity_overall_pnl = float(equity_overall_result['overall_equity_pnl']) if equity_overall_result and equity_overall_result['overall_equity_pnl'] else 0.0
        
        # Calculate overall MF P&L (total P&L from mf_holdings)
        cursor.execute("""
            SELECT 
                COALESCE(SUM(pnl), 0) as overall_mf_pnl
            FROM my_schema.mf_holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
        """)
        mf_overall_result = cursor.fetchone()
        mf_overall_pnl = float(mf_overall_result['overall_mf_pnl']) if mf_overall_result and mf_overall_result['overall_mf_pnl'] else 0.0
        
        # Calculate totals
        total_today_pnl = equity_pnl + mf_pnl + intraday_pnl
        total_overall_pnl = equity_overall_pnl + mf_overall_pnl
        
        conn.close()
        
        result = {
            "equity_pnl": equity_pnl,
            "equity_overall_pnl": equity_overall_pnl,
            "mf_pnl": mf_pnl,
            "mf_overall_pnl": mf_overall_pnl,
            "intraday_pnl": intraday_pnl,
            "total_today_pnl": total_today_pnl,
            "total_overall_pnl": total_overall_pnl
        }
        cache_set_json("today_pnl_summary", result, ttl_seconds=5)
        return cached_json_response(result, "/api/today_pnl_summary")
    except Exception as e:
        logging.error(f"Error fetching today's P&L summary: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({
            "equity_pnl": 0.0,
            "equity_overall_pnl": 0.0,
            "mf_pnl": 0.0,
            "mf_overall_pnl": 0.0,
            "intraday_pnl": 0.0,
            "total_today_pnl": 0.0,
            "total_overall_pnl": 0.0,
            "error": str(e)
        }, "/api/today_pnl_summary")

@app.get("/api/portfolio_history")
async def api_portfolio_history(days: int = Query(30, ge=1, le=365)):
    """API endpoint to get daily portfolio values for chart"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get daily equity balances
        cursor.execute("""
            SELECT 
                run_date,
                COALESCE(SUM(quantity * last_price), 0) as equity_value
            FROM my_schema.holdings
            WHERE run_date >= CURRENT_DATE - make_interval(days => %s)
            GROUP BY run_date
            ORDER BY run_date
        """, (days,))
        
        equity_data = cursor.fetchall()
        
        # Get daily MF balances
        cursor.execute("""
            SELECT 
                run_date,
                COALESCE(SUM(current_value), 0) as mf_value
            FROM my_schema.mf_holdings
            WHERE run_date >= CURRENT_DATE - make_interval(days => %s)
            GROUP BY run_date
            ORDER BY run_date
        """, (days,))
        
        mf_data = cursor.fetchall()
        
        # Create a map of dates to values
        equity_map = {str(row['run_date']): float(row['equity_value']) for row in equity_data}
        mf_map = {str(row['run_date']): float(row['mf_value']) for row in mf_data}
        
        # Get all unique dates
        all_dates = sorted(set(list(equity_map.keys()) + list(mf_map.keys())))
        
        # Build chart data
        chart_data = []
        for date in all_dates:
            equity_val = equity_map.get(date, 0.0)
            mf_val = mf_map.get(date, 0.0)
            total_val = equity_val + mf_val
            chart_data.append({
                "date": date,
                "equity": equity_val,
                "mutual_fund": mf_val,
                "total": total_val
            })
        
        conn.close()
        
        return cached_json_response({"portfolio_history": chart_data}, "/api/portfolio_history")
    except Exception as e:
        logging.error(f"Error fetching portfolio history: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"error": str(e), "portfolio_history": []}, "/api/portfolio_history")

@app.get("/api/sparkline/{trading_symbol}")
async def api_sparkline(trading_symbol: str, days: int = Query(90, ge=7, le=90)):
    """API endpoint to get sparkline data for a stock"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get recent price history for sparkline
        cursor.execute("""
            SELECT 
                price_date::date,
                price_close as price
            FROM my_schema.rt_intraday_price
            WHERE scrip_id = %s
            AND price_date::date >= CURRENT_DATE - make_interval(days => %s)
            ORDER BY price_date ASC
        """, (trading_symbol, days))
        
        sparkline_data = cursor.fetchall()
        
        # Convert to list of price values
        prices = []
        labels = []
        for row in sparkline_data:
            prices.append(float(row['price']) if row['price'] else 0.0)
            # Handle both string and date objects for price_date
            price_date = row['price_date']
            if price_date:
                if hasattr(price_date, 'strftime'):
                    labels.append(price_date.strftime('%Y-%m-%d'))
                else:
                    labels.append(str(price_date))
            else:
                labels.append('')
        
        conn.close()
        
        result = {
            "trading_symbol": trading_symbol,
            "prices": prices,
            "labels": labels,
            "min_price": min(prices) if prices else 0,
            "max_price": max(prices) if prices else 0
        }
        return cached_json_response(result, f"/api/sparkline/{trading_symbol}")
    except Exception as e:
        logging.error(f"Error fetching sparkline data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"error": str(e), "prices": [], "labels": []}, f"/api/sparkline/{trading_symbol}")

@app.post("/api/sparklines/batch")
async def api_sparklines_batch(request: Request, days: int = Query(90, ge=7, le=90)):
    """API endpoint to get sparkline data for multiple stocks in a single request"""
    try:
        body = await request.json()
        symbols = body.get('symbols', [])
        
        if not symbols or len(symbols) == 0:
            return cached_json_response({"error": "No symbols provided", "sparklines": {}}, "/api/sparklines/batch")
        
        # Limit batch size to prevent abuse
        if len(symbols) > 50:
            symbols = symbols[:50]
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get recent price history for all symbols in one query
        # Use array parameter for PostgreSQL IN clause
        placeholders = ','.join(['%s'] * len(symbols))
        cursor.execute(f"""
            SELECT 
                scrip_id,
                price_date::date,
                price_close as price
            FROM my_schema.rt_intraday_price
            WHERE scrip_id IN ({placeholders})
            AND price_date::date >= CURRENT_DATE - make_interval(days => %s)
            ORDER BY scrip_id, price_date ASC
        """, symbols + [days])
        
        sparkline_data = cursor.fetchall()
        conn.close()
        
        # Group data by symbol
        sparklines_dict = {}
        for symbol in symbols:
            sparklines_dict[symbol] = {
                "trading_symbol": symbol,
                "prices": [],
                "labels": [],
                "min_price": 0,
                "max_price": 0
            }
        
        # Process fetched data
        for row in sparkline_data:
            symbol = row['scrip_id']
            if symbol in sparklines_dict:
                price = float(row['price']) if row['price'] else 0.0
                sparklines_dict[symbol]['prices'].append(price)
                
                # Handle both string and date objects for price_date
                price_date = row['price_date']
                if price_date:
                    if hasattr(price_date, 'strftime'):
                        sparklines_dict[symbol]['labels'].append(price_date.strftime('%Y-%m-%d'))
                    else:
                        sparklines_dict[symbol]['labels'].append(str(price_date))
                else:
                    sparklines_dict[symbol]['labels'].append('')
        
        # Calculate min/max for each symbol
        for symbol, data in sparklines_dict.items():
            if data['prices']:
                data['min_price'] = min(data['prices'])
                data['max_price'] = max(data['prices'])
        
        result = {
            "sparklines": sparklines_dict
        }
        return cached_json_response(result, "/api/sparklines/batch")
    except Exception as e:
        logging.error(f"Error fetching batch sparkline data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"error": str(e), "sparklines": {}}, "/api/sparklines/batch")

@app.get("/api/candlestick/{trading_symbol}")
async def api_candlestick(trading_symbol: str, days: int = Query(30, ge=7, le=90)):
    """API endpoint to get candlestick chart data for a stock"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get OHLC data for candlestick chart
        cursor.execute("""
            SELECT 
                price_date,
                price_open,
                price_high,
                price_low,
                price_close,
                volume
            FROM my_schema.rt_intraday_price
            WHERE scrip_id = %s
            AND price_date::date >= CURRENT_DATE - make_interval(days => %s)
            ORDER BY price_date ASC
        """, (trading_symbol, days))
        
        ohlc_data = cursor.fetchall()
        
        # Convert to lists
        data = []
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        for row in ohlc_data:
            # Handle both string and date objects for price_date
            price_date = row['price_date']
            if price_date:
                if hasattr(price_date, 'strftime'):
                    date_str = price_date.strftime('%Y-%m-%d')
                else:
                    date_str = str(price_date)
            else:
                date_str = ''
            
            dates.append(date_str)
            opens.append(float(row['price_open']) if row['price_open'] else 0.0)
            highs.append(float(row['price_high']) if row['price_high'] else 0.0)
            lows.append(float(row['price_low']) if row['price_low'] else 0.0)
            closes.append(float(row['price_close']) if row['price_close'] else 0.0)
            volumes.append(float(row['volume']) if row['volume'] else 0.0)
        
        # Convert to numpy arrays
        opens_array = np.array(opens)
        highs_array = np.array(highs)
        lows_array = np.array(lows)
        closes_array = np.array(closes)
        
        # Calculate technical indicators
        if TALIB_AVAILABLE:
            try:
                logging.info(f"Calculating indicators using TA-Lib for {len(closes_array)} data points")
                sma_20 = talib.SMA(closes_array, timeperiod=20)
                sma_50 = talib.SMA(closes_array, timeperiod=50)
                sma_200 = talib.SMA(closes_array, timeperiod=200)
                supertrend, supertrend_direction = calculate_supertrend(highs_array, lows_array, closes_array)
                logging.info(f"Indicators calculated successfully")
            except Exception as e:
                logging.error(f"Error calculating indicators: {e}")
                import traceback
                logging.error(traceback.format_exc())
                # Return empty indicators
                sma_20 = np.full(len(closes_array), np.nan)
                sma_50 = np.full(len(closes_array), np.nan)
                sma_200 = np.full(len(closes_array), np.nan)
                supertrend = np.full(len(closes_array), np.nan)
                supertrend_direction = np.zeros(len(closes_array))
        else:
            # Use pandas for calculations
            logging.info(f"TA-Lib not available, using pandas for calculations")
            df = pd.DataFrame({
                'close': closes_array,
                'high': highs_array,
                'low': lows_array
            })
            sma_20 = df['close'].rolling(window=20).mean().values
            sma_50 = df['close'].rolling(window=50).mean().values
            sma_200 = df['close'].rolling(window=200).mean().values
            # Simple Supertrend calculation
            supertrend, supertrend_direction = calculate_supertrend(highs_array, lows_array, closes_array)
            logging.info(f"Indicators calculated using pandas")
        
        # Build data array with indicators (rounded to 2 decimal places)
        for i in range(len(dates)):
            data.append({
                "date": dates[i],
                "open": round(float(opens[i]), 2) if opens[i] else 0.0,
                "high": round(float(highs[i]), 2) if highs[i] else 0.0,
                "low": round(float(lows[i]), 2) if lows[i] else 0.0,
                "close": round(float(closes[i]), 2) if closes[i] else 0.0,
                "volume": volumes[i],
                "sma_20": round(float(sma_20[i]), 2) if not np.isnan(sma_20[i]) else None,
                "sma_50": round(float(sma_50[i]), 2) if not np.isnan(sma_50[i]) else None,
                "sma_200": round(float(sma_200[i]), 2) if not np.isnan(sma_200[i]) else None,
                "supertrend": round(float(supertrend[i]), 2) if not np.isnan(supertrend[i]) else None,
                "supertrend_direction": int(supertrend_direction[i]) if not np.isnan(supertrend_direction[i]) else None
            })
        
        conn.close()
        
        return {
            "trading_symbol": trading_symbol,
            "data": data
        }
    except Exception as e:
        logging.error(f"Error fetching candlestick data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e), "data": []}

@app.get("/api/today_pnl")
async def api_today_pnl():
    """API endpoint to get today's total P&L from holdings using rt_intraday_price"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get today's date in YYYY-MM-DD format
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        # Get previous trading day (assuming we can find it from rt_intraday_price)
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        if not prev_date_result or not prev_date_result['prev_date']:
            logging.warning("No previous trading day found")
            return {"total_pnl": 0.0, "message": "No previous trading day data available"}
        
        prev_date = prev_date_result['prev_date']
        
        # Calculate P&L for each holding: quantity × (today_price - prev_day_price)
        cursor.execute("""
            WITH holdings_data AS (
                SELECT 
                    h.trading_symbol,
                    h.quantity,
                    h.last_price as current_price
                FROM my_schema.holdings h
                WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ),
            today_prices AS (
                SELECT 
                    scrip_id,
                    price_close
                FROM my_schema.rt_intraday_price
                WHERE price_date = %s
            ),
            prev_prices AS (
                SELECT 
                    scrip_id,
                    price_close
                FROM my_schema.rt_intraday_price
                WHERE price_date = %s
            )
            SELECT 
                COALESCE(SUM(
                    h.quantity * (COALESCE(today_p.price_close, h.current_price) - COALESCE(prev_p.price_close, 0))
                ), 0) as total_pnl
            FROM holdings_data h
            LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
            LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
        """, (today_str, prev_date))
        
        result = cursor.fetchone()
        total_pnl = float(result['total_pnl']) if result and result['total_pnl'] is not None else 0.0
        conn.close()
        
        return {
            "total_pnl": total_pnl,
            "today_date": today_str,
            "prev_date": prev_date
        }
    except Exception as e:
        logging.error(f"Error fetching today's P&L: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e), "total_pnl": 0.0}

@app.post("/api/set_gtt")
async def api_set_gtt(
    instrument_token: str = None,
    tradingsymbol: str = None,
    quantity: int = None,
    trigger_price: float = None,
    exchange: str = kite.EXCHANGE_NSE
):
    """API endpoint to set a GTT stop-loss order"""
    try:
        from kite.KiteGTT import KiteGTTManager
        
        # Get current price from holdings
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT trading_symbol, last_price 
            FROM my_schema.holdings 
            WHERE instrument_token = %s 
            AND run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            LIMIT 1
        """, (instrument_token,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return {"error": "Holding not found"}
        
        tradingsymbol = result[0] if not tradingsymbol else tradingsymbol
        last_price = result[1]
        
        if not trigger_price:
            return {"error": "Trigger price is required"}
        
        order_price = float(trigger_price) * 0.999
        
        manager = KiteGTTManager(kite)
        gtt_response = manager.add_gtt(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            quantity=int(quantity),
            trigger_price=float(trigger_price),
            last_price=last_price,
            order_price=order_price
        )
        
        if gtt_response:
            return {
                "success": True,
                "trigger_id": gtt_response.get('trigger_id') or gtt_response.get('id'),
                "message": f"GTT set successfully for {tradingsymbol}"
            }
        else:
            return {"success": False, "error": "Failed to set GTT"}
            
    except Exception as e:
        logging.error(f"Error setting GTT: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/add_gtt_for_all")
async def api_add_gtt_for_all(stop_loss_percentage: float = 5.0):
    """API endpoint to add GTT stop-loss for all holdings"""
    try:
        from kite.KiteGTT import KiteGTTManager
        
        manager = KiteGTTManager(kite)
        results = manager.add_gtt_for_all_holdings(
            stop_loss_percentage=stop_loss_percentage,
            overwrite_existing=False
        )
        
        return {
            "success": True,
            "results": results,
            "message": f"Added {len(results['success'])} GTTs, {len(results['failed'])} failed"
        }
        
    except Exception as e:
        logging.error(f"Error adding GTT for all: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/get_all_gtts")
async def api_get_all_gtts():
    """API endpoint to get all active GTT orders"""
    try:
        from kite.KiteGTT import KiteGTTManager
        
        manager = KiteGTTManager(kite)
        gtt_list = manager.get_all_gtts()
        
        return {"gtts": gtt_list, "total": len(gtt_list)}
        
    except Exception as e:
        logging.error(f"Error fetching GTTs: {e}")
        return {"error": str(e), "gtts": []}

@app.delete("/api/cancel_gtt/{trigger_id}")
async def api_cancel_gtt(trigger_id: int):
    """API endpoint to cancel a specific GTT"""
    try:
        from kite.KiteGTT import KiteGTTManager
        
        manager = KiteGTTManager(kite)
        success = manager.cancel_gtt(trigger_id)
        
        return {"success": success}
        
    except Exception as e:
        logging.error(f"Error cancelling GTT: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/cancel_all_gtts")
async def api_cancel_all_gtts():
    """API endpoint to cancel all GTT orders"""
    try:
        from kite.KiteGTT import KiteGTTManager
        
        manager = KiteGTTManager(kite)
        results = manager.cancel_all_gtts()
        
        return {"success": True, "results": results}
        
    except Exception as e:
        logging.error(f"Error cancelling all GTTs: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/derivatives_suggestions")
async def api_derivatives_suggestions(
    instrument_token: int = Query(256265, description="Instrument token (default: 256265 for Nifty 50)"),
    analysis_date: str = Query(None, description="Analysis date in YYYY-MM-DD format (default: today)")
):
    """API endpoint to get TPO-based derivatives trading suggestions"""
    try:
        from market.CalculateTPO import PostgresDataFetcher
        from options.DerivativesTPOAnalyzer import DerivativesTPOAnalyzer
        from options.DerivativesSuggestionEngine import DerivativesSuggestionEngine
        
        # Get database configuration from Boilerplate
        db_config = {
            'host': os.getenv('PG_HOST', 'postgres'),
            'database': os.getenv('PG_DATABASE', 'mydb'),
            'user': os.getenv('PG_USER', 'postgres'),
            'password': os.getenv('PG_PASSWORD', 'postgres'),
            'port': int(os.getenv('PG_PORT', 5432))
        }
        
        # Initialize components
        db_fetcher = PostgresDataFetcher(**db_config)
        tpo_analyzer = DerivativesTPOAnalyzer(db_fetcher, instrument_token=instrument_token)
        suggestion_engine = DerivativesSuggestionEngine(tpo_analyzer)
        
        # Get current price from latest tick or quote
        current_price = None
        try:
            # Try to get current price from Kite quote
            quote = kite.quote([f"NFO:NIFTY{datetime.now().strftime('%y%b').upper()}FUT"])
            if quote:
                inst_key = list(quote.keys())[0]
                current_price = quote[inst_key].get('last_price', None)
        except Exception as e:
            logging.warning(f"Could not fetch current price from Kite: {e}")
        
        # If no price from quote, try database
        if not current_price:
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT last_price FROM my_schema.ticks
                    WHERE instrument_token = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (instrument_token,))
                result = cursor.fetchone()
                if result:
                    current_price = float(result[0])
            except Exception as e:
                logging.warning(f"Could not fetch current price from database: {e}")
            finally:
                conn.close()
        
        # Generate suggestions
        suggestions = suggestion_engine.generate_suggestions(analysis_date, current_price)
        
        return {
            "success": True,
            "analysis_date": analysis_date or datetime.now().strftime('%Y-%m-%d'),
            "instrument_token": instrument_token,
            "current_price": current_price,
            "suggestions": suggestions,
            "total_suggestions": len(suggestions)
        }
    except Exception as e:
        logging.error(f"Error generating derivatives suggestions: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "suggestions": []
        }

@app.get("/api/futures_order_flow")
async def api_futures_order_flow():
    """API endpoint to get Nifty 50 futures order flow data - shows orders for the last 2 ticks"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the last 2 distinct timestamps and all orders from those timestamps
        cursor.execute("""
            WITH recent_ticks AS (
                SELECT DISTINCT timestamp
                FROM my_schema.futures_tick_depth 
                ORDER BY timestamp DESC
                LIMIT 2
            )
            SELECT 
                ftd.timestamp,
                ftd.side,
                ftd.price,
                ftd.quantity,
                ftd.orders,
                ftd.run_date
            FROM my_schema.futures_tick_depth ftd
            WHERE ftd.timestamp IN (SELECT timestamp FROM recent_ticks)
            ORDER BY ftd.timestamp DESC, ftd.side DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        # Format the results
        order_flow_data = []
        for row in results:
            # Handle timestamp (could be string or datetime)
            timestamp_val = row[0]
            if timestamp_val:
                if hasattr(timestamp_val, 'strftime'):
                    timestamp_str = timestamp_val.strftime("%H:%M:%S")
                else:
                    timestamp_str = str(timestamp_val)[:8]  # Extract HH:MM:SS from string
            else:
                timestamp_str = ""
            
            # Handle run_date (could be string or date)
            run_date_val = row[5]
            if run_date_val:
                if hasattr(run_date_val, 'strftime'):
                    run_date_str = run_date_val.strftime("%Y-%m-%d")
                else:
                    run_date_str = str(run_date_val)
            else:
                run_date_str = ""
            
            order_flow_data.append({
                "timestamp": timestamp_str,
                "side": row[1],
                "price": float(row[2]) if row[2] else 0.0,
                "quantity": int(row[3]) if row[3] else 0,
                "orders": int(row[4]) if row[4] else 0,
                "run_date": run_date_str
            })
        
        return {
            "order_flow": order_flow_data,
            "total_orders": len(order_flow_data),
            "trading_date": order_flow_data[0]['run_date'] if order_flow_data else None
        }
    except Exception as e:
        logging.error(f"Error fetching futures order flow: {e}")
        return {"error": str(e)}


# ==================== Download Endpoints ====================

@app.get("/api/download/holdings/excel")
async def download_holdings_excel():
    """Download holdings data as Excel file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get holdings with calculated fields
        cursor.execute("""
            SELECT 
                h.trading_symbol,
                h.instrument_token,
                h.quantity,
                h.average_price,
                h.last_price,
                (h.quantity * h.average_price) as invested_amount,
                (h.quantity * COALESCE(h.last_price, rt.price_close, 0)) as current_amount,
                h.pnl,
                round(case when (h.quantity * h.average_price) != 0 then
                    (h.pnl / (h.quantity * h.average_price)) * 100
                else 0
                end::numeric, 2) as pnl_pct_change
            FROM my_schema.holdings h
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE price_date::date <= CURRENT_DATE
                ORDER BY scrip_id, price_date DESC
            ) rt ON h.trading_symbol = rt.scrip_id
            WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ORDER BY h.trading_symbol
        """)
        
        holdings = cursor.fetchall()
        
        # Get today's P&L data
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        # Create today's P&L map
        today_pnl_map = {}
        for row in holdings:
            instrument_token = row.get('instrument_token')
            if instrument_token:
                today_pnl_map[instrument_token] = {
                    'today_pnl': 0.0,
                    'pct_change': 0.0
                }
        
        if prev_date:
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT h.instrument_token, h.trading_symbol, h.quantity, h.last_price
                    FROM my_schema.holdings h
                    WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                ),
                latest_prices AS (
                    SELECT scrip_id, price_close, 
                           ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                )
                SELECT 
                    h.instrument_token,
                    COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    CASE 
                        WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                        WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                        ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                    END as today_pnl,
                    round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
                        ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
                    else 0
                    end::numeric, 2) as pct_change
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date_str))
            
            for row in cursor.fetchall():
                instrument_token = row['instrument_token']
                today_pnl_map[instrument_token] = {
                    'today_pnl': float(row['today_pnl']) if row['today_pnl'] else 0.0,
                    'pct_change': float(row['pct_change']) if row['pct_change'] else 0.0
                }
        
        # Get Prophet predictions (60-day preferred, fallback to latest)
        cursor.execute("""
            SELECT DISTINCT ON (scrip_id) scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
            FROM my_schema.prophet_predictions
            WHERE status = 'ACTIVE'
            ORDER BY scrip_id, 
                     CASE WHEN prediction_days = 60 THEN 1 ELSE 2 END,
                     run_date DESC
        """)
        
        prophet_map = {}
        for row in cursor.fetchall():
            scrip_id = row['scrip_id'].upper()
            prophet_map[scrip_id] = {
                'prediction_pct': float(row['predicted_price_change_pct']) if row['predicted_price_change_pct'] is not None else None,
                'confidence': float(row['prediction_confidence']) if row['prediction_confidence'] is not None else None,
                'prediction_days': int(row['prediction_days']) if row['prediction_days'] is not None else None
            }
        
        conn.close()
        
        # Enrich holdings data
        holdings_list = []
        for row in holdings:
            row_dict = dict(row)
            symbol = row_dict['trading_symbol']
            instrument_token = row_dict.get('instrument_token')
            
            # Add today's P&L
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'pct_change': 0.0})
            row_dict['today_pnl'] = today_pnl_info['today_pnl']
            row_dict['today_pnl_pct'] = today_pnl_info['pct_change']
            
            # Add Prophet predictions
            prophet_info = prophet_map.get(symbol.upper() if symbol else '', {})
            row_dict['ghost_prediction_pct'] = prophet_info.get('prediction_pct')
            row_dict['confidence'] = prophet_info.get('confidence')
            row_dict['prediction_days'] = prophet_info.get('prediction_days')
            
            holdings_list.append(row_dict)
        
        # Create DataFrame with proper column order
        df = pd.DataFrame(holdings_list)
        
        # Reorder columns for better readability
        column_order = [
            'trading_symbol', 'quantity', 'average_price', 'last_price',
            'invested_amount', 'current_amount', 'pnl', 'pnl_pct_change',
            'today_pnl', 'today_pnl_pct',
            'ghost_prediction_pct', 'confidence', 'prediction_days'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        # Rename columns for readability
        df = df.rename(columns={
            'trading_symbol': 'Symbol',
            'quantity': 'Qty',
            'average_price': 'Avg Price',
            'last_price': 'LTP',
            'invested_amount': 'Invested Amount',
            'current_amount': 'Current Amount',
            'pnl': 'Total P&L',
            'pnl_pct_change': 'Total P&L %',
            'today_pnl': "Today's P&L",
            'today_pnl_pct': "Today's P&L %",
            'ghost_prediction_pct': 'Ghost Prediction %',
            'confidence': 'Confidence %',
            'prediction_days': 'Prediction Days'
        })
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Holdings', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=holdings_{datetime.now().strftime('%Y%m%d')}.xlsx"}
        )
    except Exception as e:
        logging.error(f"Error generating Excel file: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}

@app.get("/api/download/holdings/csv")
async def download_holdings_csv():
    """Download holdings data as CSV file - uses same logic as Excel"""
    try:
        # Reuse the Excel download logic but return as CSV
        from io import StringIO
        
        # Use the same data fetching logic as Excel
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get holdings with calculated fields
        cursor.execute("""
            SELECT 
                h.trading_symbol,
                h.instrument_token,
                h.quantity,
                h.average_price,
                h.last_price,
                (h.quantity * h.average_price) as invested_amount,
                (h.quantity * COALESCE(h.last_price, rt.price_close, 0)) as current_amount,
                h.pnl,
                round(case when (h.quantity * h.average_price) != 0 then
                    (h.pnl / (h.quantity * h.average_price)) * 100
                else 0
                end::numeric, 2) as pnl_pct_change
            FROM my_schema.holdings h
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE price_date::date <= CURRENT_DATE
                ORDER BY scrip_id, price_date DESC
            ) rt ON h.trading_symbol = rt.scrip_id
            WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ORDER BY h.trading_symbol
        """)
        
        holdings = cursor.fetchall()
        
        # Get today's P&L data
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        # Create today's P&L map
        today_pnl_map = {}
        for row in holdings:
            instrument_token = row.get('instrument_token')
            if instrument_token:
                today_pnl_map[instrument_token] = {
                    'today_pnl': 0.0,
                    'pct_change': 0.0
                }
        
        if prev_date:
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT h.instrument_token, h.trading_symbol, h.quantity, h.last_price
                    FROM my_schema.holdings h
                    WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                ),
                latest_prices AS (
                    SELECT scrip_id, price_close, 
                           ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                )
                SELECT 
                    h.instrument_token,
                    COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    CASE 
                        WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                        WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                        ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                    END as today_pnl,
                    round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
                        ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
                    else 0
                    end::numeric, 2) as pct_change
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date_str))
            
            for row in cursor.fetchall():
                instrument_token = row['instrument_token']
                today_pnl_map[instrument_token] = {
                    'today_pnl': float(row['today_pnl']) if row['today_pnl'] else 0.0,
                    'pct_change': float(row['pct_change']) if row['pct_change'] else 0.0
                }
        
        # Get Prophet predictions (60-day preferred, fallback to latest)
        cursor.execute("""
            SELECT DISTINCT ON (scrip_id) scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
            FROM my_schema.prophet_predictions
            WHERE status = 'ACTIVE'
            ORDER BY scrip_id, 
                     CASE WHEN prediction_days = 60 THEN 1 ELSE 2 END,
                     run_date DESC
        """)
        
        prophet_map = {}
        for row in cursor.fetchall():
            scrip_id = row['scrip_id'].upper()
            prophet_map[scrip_id] = {
                'prediction_pct': float(row['predicted_price_change_pct']) if row['predicted_price_change_pct'] is not None else None,
                'confidence': float(row['prediction_confidence']) if row['prediction_confidence'] is not None else None,
                'prediction_days': int(row['prediction_days']) if row['prediction_days'] is not None else None
            }
        
        conn.close()
        
        # Enrich holdings data
        holdings_list = []
        for row in holdings:
            row_dict = dict(row)
            symbol = row_dict['trading_symbol']
            instrument_token = row_dict.get('instrument_token')
            
            # Add today's P&L
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'pct_change': 0.0})
            row_dict['today_pnl'] = today_pnl_info['today_pnl']
            row_dict['today_pnl_pct'] = today_pnl_info['pct_change']
            
            # Add Prophet predictions
            prophet_info = prophet_map.get(symbol.upper() if symbol else '', {})
            row_dict['ghost_prediction_pct'] = prophet_info.get('prediction_pct')
            row_dict['confidence'] = prophet_info.get('confidence')
            row_dict['prediction_days'] = prophet_info.get('prediction_days')
            
            holdings_list.append(row_dict)
        
        # Create DataFrame with proper column order
        df = pd.DataFrame(holdings_list)
        
        # Reorder columns for better readability
        column_order = [
            'trading_symbol', 'quantity', 'average_price', 'last_price',
            'invested_amount', 'current_amount', 'pnl', 'pnl_pct_change',
            'today_pnl', 'today_pnl_pct',
            'ghost_prediction_pct', 'confidence', 'prediction_days'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        # Rename columns for readability
        df = df.rename(columns={
            'trading_symbol': 'Symbol',
            'quantity': 'Qty',
            'average_price': 'Avg Price',
            'last_price': 'LTP',
            'invested_amount': 'Invested Amount',
            'current_amount': 'Current Amount',
            'pnl': 'Total P&L',
            'pnl_pct_change': 'Total P&L %',
            'today_pnl': "Today's P&L",
            'today_pnl_pct': "Today's P&L %",
            'ghost_prediction_pct': 'Ghost Prediction %',
            'confidence': 'Confidence %',
            'prediction_days': 'Prediction Days'
        })
        
        output = StringIO()
        df.to_csv(output, index=False)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=holdings_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    except Exception as e:
        logging.error(f"Error generating CSV file: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}

@app.get("/api/download/mf/excel")
async def download_mf_excel():
    """Download MF holdings data as Excel file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                folio, fund, tradingsymbol, quantity, average_price, last_price,
                invested_amount, current_value, pnl,
                net_change_percentage, day_change_percentage
            FROM my_schema.mf_holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
            ORDER BY fund, tradingsymbol
        """)
        
        mf_holdings = cursor.fetchall()
        conn.close()
        
        df = pd.DataFrame([dict(row) for row in mf_holdings])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='MF Holdings', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=mf_holdings_{datetime.now().strftime('%Y%m%d')}.xlsx"}
        )
    except Exception as e:
        logging.error(f"Error generating Excel file: {e}")
        return {"error": str(e)}

@app.get("/api/download/mf/csv")
async def download_mf_csv():
    """Download MF holdings data as CSV file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                folio, fund, tradingsymbol, quantity, average_price, last_price,
                invested_amount, current_value, pnl,
                net_change_percentage, day_change_percentage
            FROM my_schema.mf_holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
            ORDER BY fund, tradingsymbol
        """)
        
        mf_holdings = cursor.fetchall()
        conn.close()
        
        df = pd.DataFrame([dict(row) for row in mf_holdings])
        output = io.StringIO()
        df.to_csv(output, index=False)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=mf_holdings_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    except Exception as e:
        logging.error(f"Error generating CSV file: {e}")
        return {"error": str(e)}

@app.get("/api/download/pnl_summary/excel")
async def download_pnl_summary_excel():
    """ Downloaded complete P&L summary with equity and MF holdings as Excel file"""
    try:
        # Get all holdings data
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Fetch all data similar to api_today_pnl_summary
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date FROM my_schema.rt_intraday_price WHERE price_date < %s
        """, (today_str,))
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        equity_today, equity_overall = 0.0, 0.0
        mf_today, mf_overall = 0.0, 0.0
        intraday = 0.0
        
        if prev_date:
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT trading_symbol, quantity FROM my_schema.holdings
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (SELECT scrip_id, price_close FROM my_schema.rt_intraday_price WHERE price_date = %s),
                prev_prices AS (SELECT scrip_id, price_close FROM my_schema.rt_intraday_price WHERE price_date = %s)
                SELECT COALESCE(SUM(h.quantity * (COALESCE(today_p.price_close, 0) - COALESCE(prev_p.price_close, 0))), 0) as equity_today_pnl
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date_str))
            result = cursor.fetchone()
            equity_today = float(result['equity_today_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as equity_overall_pnl FROM my_schema.holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)")
        result = cursor.fetchone()
        equity_overall = float(result['equity_overall_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(day_change_percentage * invested_amount / 100), 0) as mf_today_pnl FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)")
        result = cursor.fetchone()
        mf_today = float(result['mf_today_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as mf_overall_pnl FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)")
        result = cursor.fetchone()
        mf_overall = float(result['mf_overall_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as intraday_pnl FROM my_schema.positions WHERE run_date = (SELECT MAX(run_date) FROM my_schema.positions) AND position_type = 'day'")
        result = cursor.fetchone()
        intraday = float(result['intraday_pnl']) if result else 0.0
        
        summary_data = [
            {'Category': 'Equity Portfolio', "Today's P&L": equity_today, 'Overall P&L': equity_overall},
            {'Category': 'Mutual Fund Portfolio', "Today's P&L": mf_today, 'Overall P&L': mf_overall},
            {'Category': 'Intraday Trades', "Today's P&L": intraday, 'Overall P&L': '-'},
            {'Category': 'Total', "Today's P&L": equity_today + mf_today + intraday, 'Overall P&L': equity_overall + mf_overall}
        ]
        
        cursor.execute("""
            SELECT trading_symbol, quantity, average_price, last_price,
                   (quantity * average_price) as invested_amount,
                   (quantity * last_price) as current_amount, pnl
            FROM my_schema.holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings) ORDER BY trading_symbol
        """)
        equity_holdings = cursor.fetchall()
        
        cursor.execute("""
            SELECT folio, fund, tradingsymbol, quantity, average_price, last_price,
                   invested_amount, current_value, pnl, net_change_percentage, day_change_percentage
            FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings) ORDER BY fund, tradingsymbol
        """)
        mf_holdings = cursor.fetchall()
        conn.close()
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='P&L Summary', index=False)
            pd.DataFrame([dict(row) for row in equity_holdings]).to_excel(writer, sheet_name='Equity Holdings', index=False)
            pd.DataFrame([dict(row) for row in mf_holdings]).to_excel(writer, sheet_name='MF Holdings', index=False)
        
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=pnl_summary_{datetime.now().strftime('%Y%m%d')}.xlsx"}
        )
    except Exception as e:
        logging.error(f"Error generating Excel file: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}

@app.get("/api/download/pnl_summary/pdf")
async def download_pnl_summary_pdf():
    """Download complete P&L summary with equity and MF holdings as PDF file"""
    try:
        # Get all holdings data (similar to Excel version)
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Fetch all data similar to download_pnl_summary_excel
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date FROM my_schema.rt_intraday_price WHERE price_date < %s
        """, (today_str,))
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        equity_today, equity_overall = 0.0, 0.0
        mf_today, mf_overall = 0.0, 0.0
        intraday = 0.0
        
        if prev_date:
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT trading_symbol, quantity FROM my_schema.holdings
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (SELECT scrip_id, price_close FROM my_schema.rt_intraday_price WHERE price_date = %s),
                prev_prices AS (SELECT scrip_id, price_close FROM my_schema.rt_intraday_price WHERE price_date = %s)
                SELECT COALESCE(SUM(h.quantity * (COALESCE(today_p.price_close, 0) - COALESCE(prev_p.price_close, 0))), 0) as equity_today_pnl
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date_str))
            result = cursor.fetchone()
            equity_today = float(result['equity_today_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as equity_overall_pnl FROM my_schema.holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)")
        result = cursor.fetchone()
        equity_overall = float(result['equity_overall_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(day_change_percentage * invested_amount / 100), 0) as mf_today_pnl FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)")
        result = cursor.fetchone()
        mf_today = float(result['mf_today_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as mf_overall_pnl FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)")
        result = cursor.fetchone()
        mf_overall = float(result['mf_overall_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as intraday_pnl FROM my_schema.positions WHERE run_date = (SELECT MAX(run_date) FROM my_schema.positions) AND position_type = 'day'")
        result = cursor.fetchone()
        intraday = float(result['intraday_pnl']) if result else 0.0
        
        # Get equity holdings
        cursor.execute("""
            SELECT trading_symbol, quantity, average_price, last_price,
                   (quantity * average_price) as invested_amount,
                   (quantity * last_price) as current_amount, pnl
            FROM my_schema.holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings) ORDER BY trading_symbol
        """)
        equity_holdings = cursor.fetchall()
        
        # Get MF holdings
        cursor.execute("""
            SELECT folio, fund, tradingsymbol, quantity, average_price, last_price,
                   invested_amount, current_value, pnl, net_change_percentage, day_change_percentage
            FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings) ORDER BY fund, tradingsymbol
        """)
        mf_holdings = cursor.fetchall()
        conn.close()
        
        # Create PDF in memory
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=A4, topMargin=0.5*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1a1a1a'), spaceAfter=12)
        
        elements = []
        elements.append(Paragraph("P&L Summary Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # P&L Summary Table
        summary_data = [
            ['Category', "Today's P&L", 'Overall P&L'],
            ['Equity Portfolio', f'Rs {equity_today:,.2f}', f'Rs {equity_overall:,.2f}'],
            ['Mutual Fund Portfolio', f'Rs {mf_today:,.2f}', f'Rs {mf_overall:,.2f}'],
            ['Intraday Trades', f'Rs {intraday:,.2f}', '-'],
            ['Total', f'Rs {equity_today + mf_today + intraday:,.2f}', f'Rs {equity_overall + mf_overall:,.2f}']
        ]
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Generate Portfolio Value Chart
        try:
            # Get portfolio history for the last 30 days using the same logic as api_portfolio_history
            cursor_hist = get_db_connection()
            cursor_history = cursor_hist.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get daily equity balances
            cursor_history.execute("""
                SELECT 
                    run_date,
                    COALESCE(SUM(quantity * last_price), 0) as equity_value
                FROM my_schema.holdings
                WHERE run_date >= CURRENT_DATE - make_interval(days => 30)
                GROUP BY run_date
                ORDER BY run_date
            """)
            equity_data = cursor_history.fetchall()
            
            # Get daily MF balances
            cursor_history.execute("""
                SELECT 
                    run_date,
                    COALESCE(SUM(current_value), 0) as mf_value
                FROM my_schema.mf_holdings
                WHERE run_date >= CURRENT_DATE - make_interval(days => 30)
                GROUP BY run_date
                ORDER BY run_date
            """)
            mf_data = cursor_history.fetchall()
            cursor_hist.close()
            
            # Create a map of dates to values
            equity_map = {str(row['run_date']): float(row['equity_value']) for row in equity_data}
            mf_map = {str(row['run_date']): float(row['mf_value']) for row in mf_data}
            
            # Get all unique dates
            all_dates = sorted(set(list(equity_map.keys()) + list(mf_map.keys())))
            
            if all_dates and len(all_dates) > 0:
                # Prepare chart data
                dates = [datetime.strptime(date, '%Y-%m-%d') for date in all_dates]
                equity_vals = [equity_map.get(date, 0.0) for date in all_dates]
                mf_vals = [mf_map.get(date, 0.0) for date in all_dates]
                total_vals = [e + m for e, m in zip(equity_vals, mf_vals)]
                
                # Create chart
                plt.figure(figsize=(8, 4))
                ax = plt.subplot(111)
                
                ax.plot(dates, equity_vals, label='Equity', color='green', linewidth=2)
                ax.plot(dates, mf_vals, label='Mutual Fund', color='blue', linewidth=2)
                ax.plot(dates, total_vals, label='Total Portfolio', color='red', linewidth=2.5)
                
                ax.set_title('Total Portfolio Value (Last 30 Days)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('Value (₹)', fontsize=10)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save chart to BytesIO
                chart_buffer = io.BytesIO()
                plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
                chart_buffer.seek(0)
                plt.close()
                
                # Add chart to PDF
                elements.append(Paragraph("Portfolio Value Trend", title_style))
                elements.append(Spacer(1, 0.2*inch))
                chart_img = Image(chart_buffer, width=6*inch, height=3*inch)
                elements.append(chart_img)
                elements.append(Spacer(1, 0.3*inch))
        except Exception as e:
            logging.error(f"Error generating portfolio chart: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Continue without chart if it fails
        
        elements.append(PageBreak())
        
        # Equity Holdings Table
        elements.append(Paragraph("Equity Holdings", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        equity_total_invested = sum(row['invested_amount'] for row in equity_holdings)
        equity_total_current = sum(row['current_amount'] for row in equity_holdings)
        equity_total_pnl = sum(row['pnl'] for row in equity_holdings)
        
        equity_data = [['Symbol', 'Qty', 'Avg Price', 'LTP', 'Invested', 'Current', 'P&L']]
        for row in equity_holdings:
            row_dict = dict(row)
            equity_data.append([
                row_dict['trading_symbol'],
                str(row_dict['quantity']),
                f"Rs {row_dict['average_price']:.2f}",
                f"Rs {row_dict['last_price']:.2f}",
                f"Rs {row_dict['invested_amount']:.2f}",
                f"Rs {row_dict['current_amount']:.2f}",
                f"Rs {row_dict['pnl']:.2f}"
            ])
        
        equity_data.append([
            'TOTAL', '', '', '',
            f'Rs {equity_total_invested:,.2f}',
            f'Rs {equity_total_current:,.2f}',
            f'Rs {equity_total_pnl:,.2f}'
        ])
        
        equity_table = Table(equity_data, repeatRows=1, colWidths=[1.2*inch, 0.5*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch, 0.9*inch])
        equity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -2), 7),
            ('FONTSIZE', (0, -1), (-1, -1), 9),
        ]))
        elements.append(equity_table)
        elements.append(Spacer(1, 0.3*inch))
        elements.append(PageBreak())
        
        # MF Holdings Table
        elements.append(Paragraph("Mutual Fund Holdings", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        mf_total_invested = sum(row['invested_amount'] for row in mf_holdings)
        mf_total_current = sum(row['current_value'] for row in mf_holdings)
        mf_total_pnl = sum(row['pnl'] for row in mf_holdings)
        
        mf_data = [['Fund', 'Symbol', 'Qty', 'Avg Price', 'LTP', 'Invested', 'Current', 'P&L']]
        for row in mf_holdings:
            row_dict = dict(row)
            mf_data.append([
                row_dict.get('fund', '')[:20],  # Truncate long fund names
                row_dict.get('tradingsymbol', ''),
                str(row_dict.get('quantity', 0)),
                f"Rs {row_dict.get('average_price', 0):.2f}",
                f"Rs {row_dict.get('last_price', 0):.2f}",
                f"Rs {row_dict.get('invested_amount', 0):.2f}",
                f"Rs {row_dict.get('current_value', 0):.2f}",
                f"Rs {row_dict.get('pnl', 0):.2f}"
            ])
        
        mf_data.append([
            'TOTAL', '', '', '', '',
            f'Rs {mf_total_invested:,.2f}',
            f'Rs {mf_total_current:,.2f}',
            f'Rs {mf_total_pnl:,.2f}'
        ])
        
        mf_table = Table(mf_data, repeatRows=1, colWidths=[1.5*inch, 1*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.9*inch, 0.9*inch, 0.8*inch])
        mf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -2), 7),
            ('FONTSIZE', (0, -1), (-1, -1), 8),
        ]))
        elements.append(mf_table)
        
        doc.build(elements)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=pnl_summary_{datetime.now().strftime('%Y%m%d')}.pdf"}
        )
    except Exception as e:
        logging.error(f"Error generating P&L Summary PDF: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}

@app.get("/api/download/holdings/pdf")
async def download_holdings_pdf():
    """Download holdings data as PDF file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get holdings with calculated fields
        cursor.execute("""
            SELECT 
                h.trading_symbol,
                h.instrument_token,
                h.quantity,
                h.average_price,
                h.last_price,
                (h.quantity * h.average_price) as invested_amount,
                (h.quantity * COALESCE(h.last_price, rt.price_close, 0)) as current_amount,
                h.pnl,
                round(case when (h.quantity * h.average_price) != 0 then
                    (h.pnl / (h.quantity * h.average_price)) * 100
                else 0
                end::numeric, 2) as pnl_pct_change
            FROM my_schema.holdings h
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE price_date::date <= CURRENT_DATE
                ORDER BY scrip_id, price_date DESC
            ) rt ON h.trading_symbol = rt.scrip_id
            WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ORDER BY h.trading_symbol
        """)
        
        holdings = cursor.fetchall()
        
        # Get today's P&L data
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        # Create today's P&L map
        today_pnl_map = {}
        for row in holdings:
            instrument_token = row.get('instrument_token')
            if instrument_token:
                today_pnl_map[instrument_token] = {
                    'today_pnl': 0.0,
                    'pct_change': 0.0
                }
        
        if prev_date:
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT h.instrument_token, h.trading_symbol, h.quantity, h.last_price
                    FROM my_schema.holdings h
                    WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                ),
                latest_prices AS (
                    SELECT scrip_id, price_close, 
                           ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                )
                SELECT 
                    h.instrument_token,
                    COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    CASE 
                        WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                        WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                        ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                    END as today_pnl,
                    round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
                        ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
                    else 0
                    end::numeric, 2) as pct_change
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date_str))
            
            for row in cursor.fetchall():
                instrument_token = row['instrument_token']
                today_pnl_map[instrument_token] = {
                    'today_pnl': float(row['today_pnl']) if row['today_pnl'] else 0.0,
                    'pct_change': float(row['pct_change']) if row['pct_change'] else 0.0
                }
        
        # Get Prophet predictions (60-day preferred, fallback to latest)
        cursor.execute("""
            SELECT DISTINCT ON (scrip_id) scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
            FROM my_schema.prophet_predictions
            WHERE status = 'ACTIVE'
            ORDER BY scrip_id, 
                     CASE WHEN prediction_days = 60 THEN 1 ELSE 2 END,
                     run_date DESC
        """)
        
        prophet_map = {}
        for row in cursor.fetchall():
            scrip_id = row['scrip_id'].upper()
            prophet_map[scrip_id] = {
                'prediction_pct': float(row['predicted_price_change_pct']) if row['predicted_price_change_pct'] is not None else None,
                'confidence': float(row['prediction_confidence']) if row['prediction_confidence'] is not None else None,
                'prediction_days': int(row['prediction_days']) if row['prediction_days'] is not None else None
            }
        
        conn.close()
        
        # Enrich holdings data
        holdings_list = []
        for row in holdings:
            row_dict = dict(row)
            symbol = row_dict['trading_symbol']
            instrument_token = row_dict.get('instrument_token')
            
            # Add today's P&L
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'pct_change': 0.0})
            row_dict['today_pnl'] = today_pnl_info['today_pnl']
            row_dict['today_pnl_pct'] = today_pnl_info['pct_change']
            
            # Add Prophet predictions
            prophet_info = prophet_map.get(symbol.upper() if symbol else '', {})
            row_dict['ghost_prediction_pct'] = prophet_info.get('prediction_pct')
            row_dict['confidence'] = prophet_info.get('confidence')
            row_dict['prediction_days'] = prophet_info.get('prediction_days')
            
            holdings_list.append(row_dict)
        
        # Create PDF in memory
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter, topMargin=0.5*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1a1a1a'), spaceAfter=12)
        
        # Calculate totals
        total_invested = sum(row['invested_amount'] for row in holdings_list)
        total_current = sum(row['current_amount'] for row in holdings_list)
        total_pnl = sum(row['pnl'] for row in holdings_list)
        total_today_pnl = sum(row['today_pnl'] for row in holdings_list)
        
        elements = []
        elements.append(Paragraph("Equity Holdings Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add summary box
        summary_data = [
            ['Total Invested:', f'Rs {total_invested:,.2f}'],
            ['Current Value:', f'Rs {total_current:,.2f}'],
            ['Total P&L:', f'Rs {total_pnl:,.2f}'],
            ["Today's P&L:", f'Rs {total_today_pnl:,.2f}']
        ]
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Create table data with updated columns
        data = [['Symbol', 'Qty', 'Avg Price', 'LTP', 'Invested', 'Current', 'P&L', 'P&L %', "Today's P&L", "Today's P&L %", 'Ghost Pred %', 'Conf %', 'Pred Days']]
        for row in holdings_list:
            row_dict = dict(row)
            pnl_pct = row_dict.get('pnl_pct_change', 0) or 0
            today_pnl = row_dict.get('today_pnl', 0) or 0
            today_pnl_pct = row_dict.get('today_pnl_pct', 0) or 0
            ghost_pred = row_dict.get('ghost_prediction_pct')
            confidence = row_dict.get('confidence')
            pred_days = row_dict.get('prediction_days')
            
            data.append([
                row_dict['trading_symbol'],
                str(row_dict['quantity']),
                f"Rs {row_dict['average_price']:.2f}",
                f"Rs {row_dict['last_price']:.2f}",
                f"Rs {row_dict['invested_amount']:.2f}",
                f"Rs {row_dict['current_amount']:.2f}",
                f"Rs {row_dict['pnl']:.2f}",
                f"{pnl_pct:.2f}%" if pnl_pct else "0.00%",
                f"Rs {today_pnl:.2f}",
                f"{today_pnl_pct:.2f}%" if today_pnl_pct else "0.00%",
                f"{ghost_pred:.2f}%" if ghost_pred is not None else "N/A",
                f"{confidence:.0f}%" if confidence is not None else "N/A",
                str(pred_days) if pred_days is not None else "N/A"
            ])
        
        # Add totals row
        data.append([
            'TOTAL',
            '',
            '',
            '',
            f'Rs {total_invested:,.2f}',
            f'Rs {total_current:,.2f}',
            f'Rs {total_pnl:,.2f}',
            '',
            f'Rs {total_today_pnl:,.2f}',
            '',
            '',
            '',
            ''
        ])
        
        # Adjust column widths for more columns (using landscape-friendly sizing)
        col_widths = [0.8*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.7*inch, 0.7*inch, 0.5*inch, 0.5*inch]
        table = Table(data, repeatRows=1, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -2), 7),
            ('FONTSIZE', (0, -1), (-1, -1), 9),
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=holdings_{datetime.now().strftime('%Y%m%d')}.pdf"}
        )
    except Exception as e:
        logging.error(f"Error generating PDF: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}

@app.get("/api/download/swing_trades/excel")
async def download_swing_trades_excel():
    """Download swing trade recommendations data as Excel file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get latest swing trade recommendations from database
        cursor.execute("""
            SELECT 
                s.scrip_id,
                s.pattern_type,
                s.entry_price,
                s.target_price,
                s.stop_loss,
                s.potential_gain_pct,
                s.risk_reward_ratio,
                s.confidence_score,
                s.holding_period_days,
                s.rationale,
                pp.predicted_price_change_pct as prophet_prediction_pct,
                pp.prediction_confidence as prophet_confidence,
                pp.prediction_days
            FROM my_schema.swing_trade_suggestions s
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
                FROM my_schema.prophet_predictions
                WHERE status = 'ACTIVE'
                ORDER BY scrip_id, 
                         CASE WHEN prediction_days = 60 THEN 1 ELSE 2 END,
                         run_date DESC
            ) pp ON s.scrip_id = pp.scrip_id
            WHERE s.run_date = (SELECT MAX(run_date) FROM my_schema.swing_trade_suggestions)
            AND s.status = 'ACTIVE'
            ORDER BY s.confidence_score DESC, s.potential_gain_pct DESC
        """)
        
        swing_trades = cursor.fetchall()
        conn.close()
        
        # Enrich data
        swing_trades_list = []
        for row in swing_trades:
            row_dict = dict(row)
            swing_trades_list.append(row_dict)
        
        # Create DataFrame
        df = pd.DataFrame(swing_trades_list)
        
        # Reorder columns for better readability
        column_order = [
            'scrip_id', 'pattern_type', 'entry_price', 'target_price', 'stop_loss',
            'potential_gain_pct', 'confidence_score', 'risk_reward_ratio',
            'holding_period_days', 'prophet_prediction_pct', 'prophet_confidence', 'prediction_days',
            'rationale'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        # Rename columns for readability
        df = df.rename(columns={
            'scrip_id': 'Symbol',
            'pattern_type': 'Pattern',
            'entry_price': 'Entry Price',
            'target_price': 'Target Price',
            'stop_loss': 'Stop Loss',
            'potential_gain_pct': 'Gain %',
            'confidence_score': 'Confidence %',
            'risk_reward_ratio': 'R/R Ratio',
            'holding_period_days': 'Holding Period (Days)',
            'prophet_prediction_pct': 'Ghost Prediction %',
            'prophet_confidence': 'Ghost Confidence %',
            'prediction_days': 'Prediction Days',
            'rationale': 'Rationale'
        })
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Swing Trades', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=swing_trades_{datetime.now().strftime('%Y%m%d')}.xlsx"}
        )
    except Exception as e:
        logging.error(f"Error generating Swing Trades Excel file: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}

@app.get("/api/download/swing_trades/pdf")
async def download_swing_trades_pdf():
    """Download swing trade recommendations data as PDF file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get latest swing trade recommendations from database
        cursor.execute("""
            SELECT 
                s.scrip_id,
                s.pattern_type,
                s.entry_price,
                s.target_price,
                s.stop_loss,
                s.potential_gain_pct,
                s.risk_reward_ratio,
                s.confidence_score,
                s.holding_period_days,
                s.rationale,
                pp.predicted_price_change_pct as prophet_prediction_pct,
                pp.prediction_confidence as prophet_confidence,
                pp.prediction_days
            FROM my_schema.swing_trade_suggestions s
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
                FROM my_schema.prophet_predictions
                WHERE status = 'ACTIVE'
                ORDER BY scrip_id, 
                         CASE WHEN prediction_days = 60 THEN 1 ELSE 2 END,
                         run_date DESC
            ) pp ON s.scrip_id = pp.scrip_id
            WHERE s.run_date = (SELECT MAX(run_date) FROM my_schema.swing_trade_suggestions)
            AND s.status = 'ACTIVE'
            ORDER BY s.confidence_score DESC, s.potential_gain_pct DESC
        """)
        
        swing_trades = cursor.fetchall()
        conn.close()
        
        # Enrich data
        swing_trades_list = []
        for row in swing_trades:
            row_dict = dict(row)
            swing_trades_list.append(row_dict)
        
        # Create PDF in memory
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter, topMargin=0.5*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1a1a1a'), spaceAfter=12)
        
        elements = []
        elements.append(Paragraph("Swing Trade Recommendations Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Calculate summary
        total_recommendations = len(swing_trades_list)
        avg_gain = sum(row.get('potential_gain_pct', 0) or 0 for row in swing_trades_list) / total_recommendations if total_recommendations > 0 else 0
        avg_confidence = sum(row.get('confidence_score', 0) or 0 for row in swing_trades_list) / total_recommendations if total_recommendations > 0 else 0
        
        # Add summary box
        summary_data = [
            ['Total Recommendations:', str(total_recommendations)],
            ['Average Gain %:', f'{avg_gain:.2f}%'],
            ['Average Confidence:', f'{avg_confidence:.2f}%']
        ]
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Create table data
        data = [['Symbol', 'Pattern', 'Entry', 'Target', 'Stop Loss', 'Gain %', 'Conf %', 'R/R', 'Days', 'Ghost %', 'Ghost Conf %', 'Pred Days']]
        for row in swing_trades_list:
            row_dict = dict(row)
            ghost_pred = row_dict.get('prophet_prediction_pct')
            ghost_conf = row_dict.get('prophet_confidence')
            pred_days = row_dict.get('prediction_days')
            
            data.append([
                row_dict.get('scrip_id', '-'),
                row_dict.get('pattern_type', '-')[:30],  # Truncate long patterns
                f"Rs {row_dict.get('entry_price', 0):.2f}" if row_dict.get('entry_price') else '-',
                f"Rs {row_dict.get('target_price', 0):.2f}" if row_dict.get('target_price') else '-',
                f"Rs {row_dict.get('stop_loss', 0):.2f}" if row_dict.get('stop_loss') else '-',
                f"{row_dict.get('potential_gain_pct', 0):.2f}%" if row_dict.get('potential_gain_pct') else '-',
                f"{row_dict.get('confidence_score', 0):.0f}%" if row_dict.get('confidence_score') else '-',
                f"{row_dict.get('risk_reward_ratio', 0):.2f}" if row_dict.get('risk_reward_ratio') else '-',
                str(row_dict.get('holding_period_days', '-')) if row_dict.get('holding_period_days') else '-',
                f"{ghost_pred:.2f}%" if ghost_pred is not None else "N/A",
                f"{ghost_conf:.0f}%" if ghost_conf is not None else "N/A",
                str(pred_days) if pred_days is not None else "N/A"
            ])
        
        # Adjust column widths for readability
        col_widths = [0.7*inch, 1.2*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.6*inch, 0.6*inch, 0.5*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.5*inch]
        table = Table(data, repeatRows=1, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=swing_trades_{datetime.now().strftime('%Y%m%d')}.pdf"}
        )
    except Exception as e:
        logging.error(f"Error generating Swing Trades PDF: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}

@app.get("/api/portfolio_hedge_analysis")
async def api_portfolio_hedge_analysis(
    target_hedge_ratio: float = Query(0.5, ge=0.0, le=1.0, description="Target hedge ratio (0.0 to 1.0)"),
    strategy_type: str = Query("all", description="Strategy type filter: all, puts, calls, collars, futures, or comma-separated")
):
    """API endpoint to get portfolio hedge analysis with multiple strategies"""
    try:
        from holdings.PortfolioHedgeAnalyzer import PortfolioHedgeAnalyzer
        
        db_config = {
            'host': os.getenv('PG_HOST', 'postgres'),
            'database': os.getenv('PG_DATABASE', 'mydb'),
            'user': os.getenv('PG_USER', 'postgres'),
            'password': os.getenv('PG_PASSWORD', 'postgres'),
            'port': int(os.getenv('PG_PORT', 5432))
        }
        
        analyzer = PortfolioHedgeAnalyzer(db_config)
        
        # Calculate portfolio beta
        beta_result = analyzer.calculate_portfolio_beta()
        
        # Get hedge suggestions with strategy filtering
        hedge_suggestions = analyzer.suggest_hedge(target_hedge_ratio=target_hedge_ratio, strategy_type=strategy_type)

        # Ensure diagnostics are always included, even if missing
        if 'diagnostics' not in hedge_suggestions:
            logging.warning("hedge_suggestions missing diagnostics, adding default diagnostics")
            # Get holdings for diagnostics if missing
            try:
                holdings = analyzer.get_current_holdings()
                equity_count = len(holdings[holdings['holding_type'] == 'EQUITY']) if not holdings.empty and 'holding_type' in holdings.columns else 0
                mf_count = len(holdings[holdings['holding_type'] == 'MF']) if not holdings.empty and 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0
                equity_value = float(holdings[holdings['holding_type'] == 'EQUITY']['current_value'].sum()) if not holdings.empty and 'holding_type' in holdings.columns and 'EQUITY' in holdings['holding_type'].values else 0.0
                mf_value = float(holdings[holdings['holding_type'] == 'MF']['current_value'].sum()) if not holdings.empty and 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0.0
            except Exception as e:
                logging.error(f"Failed to compute diagnostics fallback: {e}")
                equity_count = mf_count = 0
                equity_value = mf_value = 0.0

            hedge_suggestions['diagnostics'] = {
                'holdings_count': equity_count + mf_count,
                'equity_count': equity_count,
                'mf_count': mf_count,
                'portfolio_value': float(beta_result.get('portfolio_value', equity_value + mf_value)),
                'equity_value': float(equity_value),
                'mf_value': float(mf_value),
                'beta': float(beta_result.get('beta', 0.0)),
                'correlation': float(beta_result.get('correlation', 0.0)),
                'nifty_price': 0.0,
                'expiries_available': 0,
                'futures_strategies': 0,
                'puts_strategies': 0,
                'calls_strategies': 0,
                'collars_strategies': 0
            }
        
        # Ensure diagnostics are always included, even if missing
        if 'diagnostics' not in hedge_suggestions:
            logging.warning("hedge_suggestions missing diagnostics, adding default diagnostics")
            # Get holdings for diagnostics if missing
            try:
                holdings = analyzer.get_current_holdings()
                equity_count = len(holdings[holdings['holding_type'] == 'EQUITY']) if not holdings.empty and 'holding_type' in holdings.columns else 0
                mf_count = len(holdings[holdings['holding_type'] == 'MF']) if not holdings.empty and 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0
                equity_value = float(holdings[holdings['holding_type'] == 'EQUITY']['current_value'].sum()) if not holdings.empty and 'holding_type' in holdings.columns and 'EQUITY' in holdings['holding_type'].values else 0.0
                mf_value = float(holdings[holdings['holding_type'] == 'MF']['current_value'].sum()) if not holdings.empty and 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0.0
            except:
                equity_count = mf_count = 0
                equity_value = mf_value = 0.0
            
            hedge_suggestions['diagnostics'] = {
                'holdings_count': equity_count + mf_count,
                'equity_count': equity_count,
                'mf_count': mf_count,
                'portfolio_value': float(beta_result.get('portfolio_value', equity_value + mf_value)),
                'equity_value': float(equity_value),
                'mf_value': float(mf_value),
                'beta': float(beta_result.get('beta', 0.0)),
                'correlation': float(beta_result.get('correlation', 0.0)),
                'nifty_price': 0.0,
                'expiries_available': 0,
                'futures_strategies': 0,
                'puts_strategies': 0,
                'calls_strategies': 0,
                'collars_strategies': 0
            }
        
        # Calculate VaR
        var_result = analyzer.calculate_var()
        
        return {
            "success": True,
            "portfolio_metrics": beta_result,
            "hedge_suggestions": hedge_suggestions,
            "var_metrics": var_result,
            "target_hedge_ratio": target_hedge_ratio,
            "strategy_type": strategy_type
        }
    except Exception as e:
        logging.error(f"Error generating portfolio hedge analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/derivatives_history")
async def api_derivatives_history(
    start: str = Query(None, description="Start date YYYY-MM-DD"),
    end: str = Query(None, description="End date YYYY-MM-DD"),
    strategy: str = Query(None, description="Strategy type filter"),
    instrument: str = Query(None, description="Instrument (tradingsymbol) filter"),
    limit: int = Query(200, ge=1, le=1000)
):
    """Return saved derivative suggestions history with optional filters."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        where_clauses = []
        params = []
        if start:
            where_clauses.append("generated_at::date >= %s")
            params.append(start)
        if end:
            where_clauses.append("generated_at::date <= %s")
            params.append(end)
        if strategy:
            where_clauses.append("strategy_type = %s")
            params.append(strategy.upper())
        if instrument:
            where_clauses.append("instrument = %s")
            params.append(instrument)

        sql = f"""
            SELECT id, generated_at, analysis_date, source, strategy_type, strategy_name,
                   instrument, instrument_token, direction, quantity, lot_size, entry_price,
                   strike_price, expiry, total_premium, total_premium_income, margin_required,
                   hedge_value, coverage_percentage, portfolio_value, beta, rationale,
                   tpo_context, diagnostics
            FROM my_schema.derivative_suggestions
            {('WHERE ' + ' AND '.join(where_clauses)) if where_clauses else ''}
            ORDER BY generated_at DESC
            LIMIT %s
        """
        params.append(limit)
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"success": True, "rows": rows}
    except Exception as e:
        logging.error(f"Error fetching derivative suggestions history: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.get("/api/options_scanner")
async def api_options_scanner(
    expiry: str = Query(None, description="Expiry date in YYYY-MM-DD format"),
    strike_range_min: float = Query(None, description="Minimum strike price"),
    strike_range_max: float = Query(None, description="Maximum strike price"),
    option_type: str = Query(None, description="Option type: CE or PE"),
    strategy_type: str = Query("covered_call", description="Strategy type: covered_call, cash_secured_put, iron_condor, strangle, straddle, vertical_spread"),
    min_iv_rank: float = Query(50.0, description="Minimum IV Rank (0-100)"),
    max_iv_rank: float = Query(100.0, description="Maximum IV Rank (0-100)"),
    min_liquidity_score: float = Query(0.5, description="Minimum liquidity score (0-1)"),
    min_volume: int = Query(0, description="Minimum daily volume"),
    min_oi: int = Query(0, description="Minimum open interest"),
    max_days_to_expiry: int = Query(60, description="Maximum days to expiry"),
    min_days_to_expiry: int = Query(7, description="Minimum days to expiry"),
    min_delta: float = Query(None, description="Minimum delta"),
    max_delta: float = Query(None, description="Maximum delta"),
    limit: int = Query(50, description="Maximum number of results to return")
):
    """API endpoint for advanced options chain scanning with Greeks and IV Rank"""
    try:
        from options.OptionsScanner import OptionsScanner
        from datetime import datetime, date
        
        scanner = OptionsScanner()
        
        # Parse expiry date
        expiry_date = None
        if expiry:
            try:
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            except:
                pass
        
        # Get strike range
        strike_range = None
        if strike_range_min is not None and strike_range_max is not None:
            strike_range = (strike_range_min, strike_range_max)
        
        # Scan options chain
        try:
            candidates = scanner.scan_options_chain(
                expiry=expiry_date,
                strike_range=strike_range,
                option_type=option_type,
                strategy_type=strategy_type,
                min_iv_rank=min_iv_rank,
                max_iv_rank=max_iv_rank,
                min_liquidity_score=min_liquidity_score,
                min_volume=min_volume,
                min_oi=min_oi,
                max_days_to_expiry=max_days_to_expiry,
                min_days_to_expiry=min_days_to_expiry,
                min_delta=min_delta,
                max_delta=max_delta,
                current_spot=None  # Auto-fetch
            )
            
            # Ensure candidates is a list
            if candidates is None:
                candidates = []
            elif not isinstance(candidates, list):
                candidates = list(candidates) if hasattr(candidates, '__iter__') else []
            
            # Sort by overall score if available, otherwise keep original order
            if candidates and isinstance(candidates[0], dict) and 'overall_score' in candidates[0]:
                candidates.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
            
            # Return all candidates (frontend will limit to top 5)
            return {
                "success": True,
                "candidates": candidates,
                "total_candidates": len(candidates),
                "filters_applied": {
                    "expiry": expiry,
                    "strike_range": strike_range,
                    "option_type": option_type,
                    "strategy_type": strategy_type,
                    "min_iv_rank": min_iv_rank,
                    "max_iv_rank": max_iv_rank,
                    "min_liquidity_score": min_liquidity_score,
                    "min_volume": min_volume,
                    "min_oi": min_oi,
                    "max_days_to_expiry": max_days_to_expiry,
                    "min_days_to_expiry": min_days_to_expiry,
                    "min_delta": min_delta,
                    "max_delta": max_delta
                }
            }
        except Exception as scan_error:
            logging.error(f"Error in options chain scanning: {scan_error}")
            import traceback
            logging.error(traceback.format_exc())
            # Return graceful error message
            error_msg = f"Unable to scan options chain. Please try again or adjust your filters."
            if "connection" in str(scan_error).lower() or "timeout" in str(scan_error).lower():
                error_msg = "Connection error while fetching options data. Please check your internet connection and try again."
            elif "authentication" in str(scan_error).lower() or "token" in str(scan_error).lower():
                error_msg = "Authentication error. Please refresh your session and try again."
            
            return {
                "success": False,
                "error": error_msg,
                "candidates": []
            }
    except Exception as e:
        logging.error(f"Error scanning options chain: {e}")
        import traceback
        logging.error(traceback.format_exc())
        # Return user-friendly error message
        error_msg = "An error occurred while scanning options. Please try again later."
        if "connection" in str(e).lower():
            error_msg = "Unable to connect to the options data service. Please check your connection."
        elif "timeout" in str(e).lower():
            error_msg = "Request timed out. Please try again with fewer filters or later."
        
        return {
            "success": False,
            "error": error_msg,
            "candidates": []
        }

@app.get("/api/options_chain")
async def api_options_chain(
    expiry: str = Query(None, description="Expiry date in YYYY-MM-DD format (default: current week expiry)"),
    strike_range_min: float = Query(None, description="Minimum strike price"),
    strike_range_max: float = Query(None, description="Maximum strike price"),
    option_type: str = Query(None, description="Option type: CE or PE"),
    min_volume: int = Query(0, description="Minimum volume filter"),
    min_oi: int = Query(0, description="Minimum open interest filter")
):
    """API endpoint to get options chain for NIFTY"""
    try:
        from options.OptionsDataFetcher import OptionsDataFetcher
        from datetime import datetime, date
        
        fetcher = OptionsDataFetcher()
        
        # Parse expiry date
        expiry_date = None
        if expiry:
            try:
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            except:
                pass
        
        # Get strike range
        strike_range = None
        if strike_range_min is not None and strike_range_max is not None:
            strike_range = (strike_range_min, strike_range_max)
        
        # Get options chain
        options_chain = fetcher.get_options_chain(
            expiry=expiry_date,
            strike_range=strike_range,
            option_type=option_type,
            min_volume=min_volume,
            min_oi=min_oi
        )
        
        if options_chain.empty:
            return {
                "success": True,
                "options_chain": [],
                "total_options": 0,
                "message": "No options data found for given criteria"
            }
        
        # Convert DataFrame to list of dicts
        options_list = options_chain.to_dict('records')
        
        # Convert datetime and date objects to strings
        for opt in options_list:
            if isinstance(opt.get('timestamp'), datetime):
                opt['timestamp'] = opt['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(opt.get('expiry'), date):
                opt['expiry'] = opt['expiry'].strftime('%Y-%m-%d')
        
        return {
            "success": True,
            "options_chain": options_list,
            "total_options": len(options_list),
            "filters": {
                "expiry": expiry,
                "strike_range": strike_range,
                "option_type": option_type,
                "min_volume": min_volume,
                "min_oi": min_oi
            }
        }
    except Exception as e:
        logging.error(f"Error fetching options chain: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/options_data")
async def api_options_data(
    instrument_token: int = Query(..., description="Option instrument token")
):
    """API endpoint to get real-time options data for specific instrument"""
    try:
        from options.OptionsDataFetcher import OptionsDataFetcher
        from datetime import datetime, date
        
        fetcher = OptionsDataFetcher()
        
        option_data = fetcher.get_option_quote(instrument_token)
        
        if not option_data:
            return {
                "success": False,
                "error": f"No data found for instrument_token {instrument_token}"
            }
        
        # Convert datetime and date objects to strings
        if isinstance(option_data.get('timestamp'), datetime):
            option_data['timestamp'] = option_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(option_data.get('expiry'), date):
            option_data['expiry'] = option_data['expiry'].strftime('%Y-%m-%d')
        
        return {
            "success": True,
            "option_data": option_data
        }
    except Exception as e:
        logging.error(f"Error fetching options data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/options_latest")
async def api_options_latest(
    limit: int = Query(5, description="Number of latest options to return (default: 5)")
):
    """API endpoint to get latest N options based on timestamp"""
    try:
        from options.OptionsDataFetcher import OptionsDataFetcher
        from datetime import datetime, date
        
        fetcher = OptionsDataFetcher()
        
        # Get latest options by timestamp
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT DISTINCT ON (instrument_token)
                instrument_token,
                tradingsymbol,
                strike_price,
                option_type,
                expiry,
                last_price,
                volume,
                oi,
                average_price,
                timestamp,
                buy_quantity,
                sell_quantity
            FROM my_schema.options_ticks
            WHERE run_date = CURRENT_DATE
            ORDER BY instrument_token, timestamp DESC
        """)
        
        # Get all unique options, then sort by timestamp and limit
        all_rows = cursor.fetchall()
        
        # Sort by timestamp descending and take top N
        sorted_rows = sorted(all_rows, key=lambda x: x['timestamp'] if x.get('timestamp') else datetime.min, reverse=True)[:limit]
        
        conn.close()
        
        if not sorted_rows:
            return {
                "success": True,
                "options": [],
                "total_options": 0,
                "message": "No options data found for today"
            }
        
        # Convert to list of dicts and format dates
        options_list = []
        for row in sorted_rows:
            opt = dict(row)
            if isinstance(opt.get('timestamp'), datetime):
                opt['timestamp'] = opt['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(opt.get('expiry'), date):
                opt['expiry'] = opt['expiry'].strftime('%Y-%m-%d')
            options_list.append(opt)
        
        return cached_json_response({
            "success": True,
            "options": options_list,
            "total_options": len(options_list)
        }, "/api/options_latest")
    except Exception as e:
        logging.error(f"Error fetching latest options: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({
            "success": False,
            "error": str(e)
        }, "/api/options_latest")

@app.get("/api/options_oi_analysis")
async def api_options_oi_analysis(
    expiry: str = Query(None, description="Expiry date in YYYY-MM-DD format (default: latest expiry)"),
    option_type: str = Query(None, description="Option type: 'CE' or 'PE' (default: both)"),
    analysis_date: str = Query(None, description="Analysis date in YYYY-MM-DD format (default: today)"),
    top_n: int = Query(50, description="Number of top strikes to display (default: 50)"),
    include_chart: bool = Query(True, description="Include chart image in response (default: True)")
):
    """
    API endpoint for Options Open Interest analysis
    Joins options_ticks, options_tick_ohlc, and options_tick_depth tables
    to analyze OI distribution by strike price
    """
    try:
        from options.OptionsOIAnalyzer import OptionsOIAnalyzer
        from datetime import datetime, date
        
        analyzer = OptionsOIAnalyzer()
        
        # Parse dates
        expiry_date = None
        if expiry:
            try:
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid expiry date format. Use YYYY-MM-DD"
                }
        
        analysis_date_obj = None
        if analysis_date:
            try:
                analysis_date_obj = datetime.strptime(analysis_date, '%Y-%m-%d').date()
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid analysis_date format. Use YYYY-MM-DD"
                }
        
        # Validate option_type
        if option_type and option_type not in ['CE', 'PE']:
            return {
                "success": False,
                "error": "option_type must be 'CE' or 'PE'"
            }
        
        # Generate analysis report
        if include_chart:
            report = analyzer.generate_oi_analysis_report(
                expiry=expiry_date,
                analysis_date=analysis_date_obj
            )
        else:
            # Just get data without chart
            oi_data = analyzer.get_oi_by_strike(
                expiry=expiry_date,
                option_type=option_type,
                analysis_date=analysis_date_obj,
                include_ohlc=True
            )
            summary = analyzer.get_oi_summary(
                expiry=expiry_date,
                analysis_date=analysis_date_obj
            )
            
            # Convert DataFrame to dict
            oi_data_dict = oi_data.to_dict('records') if not oi_data.empty else []
            
            # Convert numpy types
            import numpy as np
            import pandas as pd
            for record in oi_data_dict:
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.int64)):
                        record[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        record[key] = float(value) if not pd.isna(value) else None
                    elif isinstance(value, pd.Timestamp):
                        record[key] = value.isoformat()
                    elif pd.isna(value):
                        record[key] = None
            
            report = {
                'success': True,
                'analysis_date': (analysis_date_obj or date.today()).isoformat(),
                'expiry': expiry_date.isoformat() if expiry_date else None,
                'summary': summary,
                'oi_data': oi_data_dict,
                'total_records': len(oi_data),
                'chart_image': None
            }
        
        return report
        
    except Exception as e:
        logging.error(f"Error generating OI analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/margin/calculate")
async def api_margin_calculate(
    instrument_token: int = Query(..., description="Instrument token"),
    quantity: int = Query(..., description="Quantity in lots"),
    entry_price: float = Query(..., description="Entry price"),
    instrument_type: str = Query("FUTURES", description="FUTURES or OPTIONS"),
    product: str = Query("MIS", description="MIS/CNC/NRML"),
    strike_price: float = Query(None, description="Strike price (for options)"),
    premium: float = Query(None, description="Premium per share (for options)"),
    option_type: str = Query(None, description="CE or PE (for options)"),
    is_long: bool = Query(True, description="Long or Short position")
):
    """API endpoint to calculate margin requirements"""
    try:
        from common.MarginCalculator import MarginCalculator
        
        calculator = MarginCalculator()
        
        if instrument_type.upper() == "FUTURES":
            result = calculator.calculate_futures_margin(
                instrument_token=instrument_token,
                quantity=quantity,
                entry_price=entry_price,
                product=product
            )
        elif instrument_type.upper() == "OPTIONS":
            if not strike_price or not premium or not option_type:
                return {
                    "success": False,
                    "error": "strike_price, premium, and option_type are required for options"
                }
            result = calculator.calculate_options_margin(
                instrument_token=instrument_token,
                quantity=quantity,
                strike_price=strike_price,
                premium=premium,
                option_type=option_type,
                is_long=is_long,
                product=product
            )
        else:
            return {
                "success": False,
                "error": "instrument_type must be FUTURES or OPTIONS"
            }
        
        # Check margin sufficiency
        required_margin = result.get('total_margin', result.get('total_required', 0))
        margin_check = calculator.check_margin_sufficiency(required_margin)
        
        return {
            "success": True,
            "margin_calculation": result,
            "margin_check": margin_check,
            "available_margin": calculator.get_available_margin()
        }
    except Exception as e:
        logging.error(f"Error calculating margin: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/margin/available")
async def api_margin_available():
    """API endpoint to get available margin"""
    try:
        from common.MarginCalculator import MarginCalculator
        
        calculator = MarginCalculator()
        available = calculator.get_available_margin()
        
        return {
            "success": True,
            "margin_data": available
        }
    except Exception as e:
        logging.error(f"Error fetching available margin: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/download/mf/pdf")
async def download_mf_pdf():
    """Download MF holdings data as PDF file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT folio, fund, tradingsymbol, invested_amount, current_value, pnl, net_change_percentage, day_change_percentage
            FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings) ORDER BY fund, tradingsymbol
        """)
        
        mf_holdings = cursor.fetchall()
        conn.close()
        
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1a1a1a'), spaceAfter=12)
        
        elements = []
        elements.append(Paragraph("Mutual Fund Holdings Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        data = [['Fund', 'Symbol', 'Invested', 'Current', 'P&L', 'Net %', 'Day %']]
        for row in mf_holdings:
            row_dict = dict(row)
            data.append([
                row_dict['fund'] or row_dict['tradingsymbol'],
                row_dict['tradingsymbol'],
                f"Rs {row_dict['invested_amount']:.2f}",
                f"Rs {row_dict['current_value']:.2f}",
                f"Rs {row_dict['pnl']:.2f}",
                f"{row_dict['net_change_percentage']:.2f}%",
                f"{row_dict['day_change_percentage']:.2f}%"
            ])
        
        table = Table(data, repeatRows=1, colWidths=[2*inch, 1.2*inch, 1.1*inch, 1.1*inch, 0.9*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
        ]))
        
        elements.append(table)
        doc.build(elements)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=mf_holdings_{datetime.now().strftime('%Y%m%d')}.pdf"}
        )
    except Exception as e:
        logging.error(f"Error generating PDF: {e}")
        return {"error": str(e)}