from common.Boilerplate import *
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Request, Query, Form, File, UploadFile, Body, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from typing import Optional, List
import os
from datetime import datetime, timedelta, date
import json
from api.services.holdings_service import HoldingsService

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
from psycopg2.extras import RealDictCursor
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

# Function to get latest supertrend value for a stock
def get_latest_supertrend(scrip_id: str, conn=None):
    """Get the latest supertrend value, direction, corresponding close price, and days below supertrend
    Uses the same logic as the candlestick endpoint
    Returns: (supertrend_value, direction, close_price, days_below_supertrend) or None if error
    direction: -1 if price is below supertrend, 1 if price is above supertrend
    days_below_supertrend: number of consecutive days below supertrend in the latest downtrend
    """
    print(f"---XXX--- get_latest_supertrend called for {scrip_id}")
    try:
        # Always create a new connection to avoid connection issues
        # The passed conn might be closed or in use by other queries
        print(f"---XXX--- Step 1: Creating new connection for {scrip_id}")
        conn = get_db_connection()
        should_close = True
        print(f"---XXX--- Step 2: Created new connection for {scrip_id}")
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        print(f"---XXX--- Step 3: Created cursor for {scrip_id}")
        
        # Get OHLC data for candlestick - same query as api_candlestick endpoint
        # Get last 90 days of data ordered by date ASC (oldest to newest) to properly calculate days below supertrend
        print(f"---XXX--- Step 4: Executing query for {scrip_id}")
        cursor.execute("""
            SELECT 
                price_high,
                price_low,
                price_close,
                price_date
            FROM my_schema.rt_intraday_price
            WHERE scrip_id = %s
            AND price_date::date >= CURRENT_DATE - make_interval(days => 90)
            AND price_high IS NOT NULL
            AND price_low IS NOT NULL
            AND price_close IS NOT NULL
            ORDER BY price_date ASC
        """, (scrip_id,))
        
        print(f"---XXX--- Step 5: Fetching rows for {scrip_id}")
        rows = cursor.fetchall()
        cursor.close()
        print(f"---XXX--- Step 6: Closed cursor for {scrip_id}")
        
        if should_close:
            conn.close()
            print(f"---XXX--- Step 7: Closed connection for {scrip_id}")
        
        print(f"---XXX--- Found {len(rows) if rows else 0} rows for {scrip_id}")
        
        if not rows or len(rows) < 14:
            print(f"---XXX--- Not enough data for {scrip_id}: {len(rows) if rows else 0} rows")
            logging.debug(f"Not enough data for supertrend calculation for {scrip_id}: {len(rows) if rows else 0} rows")
            return None
        
        # Extract arrays - same as candlestick endpoint
        highs = []
        lows = []
        closes = []
        dates = []
        
        for row in rows:
            highs.append(float(row['price_high']) if row['price_high'] else 0.0)
            lows.append(float(row['price_low']) if row['price_low'] else 0.0)
            closes.append(float(row['price_close']) if row['price_close'] else 0.0)
            dates.append(row['price_date'])
        
        # Convert to numpy arrays
        highs_array = np.array(highs)
        lows_array = np.array(lows)
        closes_array = np.array(closes)
        
        # Calculate supertrend - same as candlestick endpoint
        try:
            supertrend, supertrend_direction = calculate_supertrend(highs_array, lows_array, closes_array)
            print('---XXX--- Supertrend')
            print(supertrend)
        except Exception as calc_error:
            logging.warning(f"Error calculating supertrend for {scrip_id}: {calc_error}")
            import traceback
            logging.warning(traceback.format_exc())
            return None
        
        # Get the latest non-NaN supertrend value (last element in array)
        # Same logic as candlestick endpoint - get last value
        latest_supertrend = None
        latest_direction = None
        latest_close = None
        latest_index = None
        
        # Find the last non-NaN value
        for i in range(len(supertrend) - 1, -1, -1):
            if not np.isnan(supertrend[i]):
                latest_supertrend = float(supertrend[i])
                latest_direction = int(supertrend_direction[i])
                latest_close = float(closes_array[i])
                latest_index = i
                break
        
        if latest_supertrend is None:
            logging.debug(f"No valid supertrend value found for {scrip_id} (all NaN)")
            return None
        
        # Calculate days below supertrend (only for latest downtrend)
        # Count consecutive calendar days with direction = -1 (below supertrend) from the latest day backwards
        # This looks back at least 90 days or from when supertrend data is available
        days_below_supertrend = 0
        if latest_direction == -1:  # Currently below supertrend
            # Count backwards from latest_index, counting actual calendar days
            # Normalize dates to date objects for proper comparison
            from datetime import date as date_type
            prev_date = None
            for i in range(latest_index, -1, -1):
                if not np.isnan(supertrend_direction[i]) and int(supertrend_direction[i]) == -1:
                    current_date_obj = dates[i]
                    # Normalize to date if it's a datetime
                    if hasattr(current_date_obj, 'date'):
                        current_date_obj = current_date_obj.date()
                    elif isinstance(current_date_obj, str):
                        from datetime import datetime
                        try:
                            current_date_obj = datetime.strptime(current_date_obj.split()[0], '%Y-%m-%d').date()
                        except:
                            current_date_obj = current_date_obj
                    
                    # If this is the first iteration or dates are different, count as a day
                    if prev_date is None or current_date_obj != prev_date:
                        days_below_supertrend += 1
                        prev_date = current_date_obj
                else:
                    break  # Stop counting when we hit a day above supertrend
        
        logging.debug(f"Supertrend calculated for {scrip_id}: value={latest_supertrend}, direction={latest_direction}, close={latest_close}, days_below={days_below_supertrend}")
        return (latest_supertrend, latest_direction, latest_close, days_below_supertrend)
        
    except Exception as e:
        print(f"---XXX--- Exception in get_latest_supertrend for {scrip_id}: {e}")
        import traceback
        print(f"---XXX--- Traceback: {traceback.format_exc()}")
        logging.debug(f"Error getting supertrend for {scrip_id}: {e}")
        logging.debug(traceback.format_exc())
        return None

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


# Function to start KiteWS automatically
def start_kitews_automatically():
    """Helper function to automatically start KiteWS after token fetch"""
    try:
        import subprocess
        import os
        import sys
        from datetime import datetime, timedelta
        from pytz import timezone
        
        # Check if KiteWS is already running
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'KiteWS.py'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                logging.info("KiteWS is already running, skipping auto-start")
                return True
        except Exception as e:
            # pgrep might not be available, continue anyway
            logging.debug(f"Could not check if KiteWS is running: {e}")
        
        # Start KiteWS using the wrapper script
        kitews_wrapper_path = os.path.join(os.path.dirname(__file__), 'kite', 'StartKiteWS.py')
        
        if not os.path.exists(kitews_wrapper_path):
            logging.warning(f"StartKiteWS.py not found at {kitews_wrapper_path}, skipping auto-start")
            return False
        
        # Start the process in the background
        process = subprocess.Popen(
            [sys.executable, kitews_wrapper_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(kitews_wrapper_path)),
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        ist = timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        target_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # If current time is past 3:30 PM, set target to next day
        if current_time >= target_time:
            target_time = target_time + timedelta(days=1)
        
        logging.info(f"Auto-started KiteWS process (PID: {process.pid}) at {current_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
        logging.info(f"KiteWS will run until {target_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
        return True
        
    except Exception as e:
        logging.error(f"Error auto-starting KiteWS: {e}")
        import traceback
        logging.error(traceback.format_exc())
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

        # Auto-start KiteWS after successful token generation
        logging.info("Auto-starting KiteWS after successful token fetch...")
        start_kitews_automatically()

        return access_token
    except Exception as e:
        logging.error(f"Failed to generate new access token: {e}")
        return None


# FastAPI app for capturing request_token
app = FastAPI()

# Include API routers
from api.routers import holdings_router
app.include_router(holdings_router)

# Initialize services
holdings_service = HoldingsService()

# Template configuration
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Response compression to reduce payload size
app.add_middleware(GZipMiddleware, minimum_size=500)

# Global ANALYSIS_DATE configuration - can be set via API or environment
ANALYSIS_DATE = None  # Set to None for current date, or specify date like '2025-10-20'

# Global SHOW_HOLDINGS configuration - controls visibility of holdings sections
# Only True if explicitly set to "True" in .env, otherwise False
SHOW_HOLDINGS = os.getenv("SHOW_HOLDINGS", "False").lower() == "true"

# Global tab visibility configurations - controls visibility of specific tabs
# Only True if explicitly set to "True" in .env, otherwise False
SHOW_STOCKS_TAB = os.getenv("SHOW_STOCKS_TAB", "False").lower() == "true"
SHOW_OPTIONS_TAB = os.getenv("SHOW_OPTIONS_TAB", "False").lower() == "true"
SHOW_UTILITIES_TAB = os.getenv("SHOW_UTILITIES_TAB", "False").lower() == "true"
SHOW_FUNDAMENTALS_TAB = os.getenv("SHOW_FUNDAMENTALS_TAB", "False").lower() == "true"

# Debug logging for tab visibility settings
logging.info(f"Tab visibility settings - SHOW_STOCKS_TAB: {SHOW_STOCKS_TAB}, SHOW_OPTIONS_TAB: {SHOW_OPTIONS_TAB}, SHOW_UTILITIES_TAB: {SHOW_UTILITIES_TAB}, SHOW_FUNDAMENTALS_TAB: {SHOW_FUNDAMENTALS_TAB}")

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

# Function to check if KiteWS is running by checking for recent database data
def check_kitews_running_by_data(minutes_threshold: int = 3) -> bool:
    """
    Check if KiteWS is running by checking if there's recent data in the database.
    
    Args:
        minutes_threshold: Number of minutes to check back for recent data (default: 3)
        
    Returns:
        True if recent data exists (indicating KiteWS is likely running), False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check for ticks in the last N minutes
        # Using string formatting for interval since PostgreSQL doesn't support parameterized intervals
        cursor.execute(f"""
            SELECT COUNT(*) FROM my_schema.ticks 
            WHERE timestamp > NOW() - INTERVAL '{minutes_threshold} minutes'
            AND instrument_token IN (256265, 52168, 37054)
        """)
        
        recent_ticks = cursor.fetchone()[0]
        conn.close()
        
        is_running = recent_ticks > 0
        logging.debug(f"KiteWS data check: {recent_ticks} ticks in last {minutes_threshold} minutes, is_running={is_running}")
        
        return is_running
        
    except Exception as e:
        logging.error(f"Error checking KiteWS data status: {e}")
        return False

# Function to check and auto-start KiteWS if not running
def check_and_auto_start_kitews():
    """
    Check if KiteWS is running (by checking database for recent data) and auto-start it if not.
    This function checks both process status and database data to ensure KiteWS is actually working.
    
    Returns:
        dict with status information
    """
    try:
        import subprocess
        from datetime import datetime
        from pytz import timezone
        
        ist = timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Only check during market hours (9:00 AM to 3:30 PM IST)
        if current_hour < 9 or (current_hour == 15 and current_minute > 30) or current_hour > 15:
            logging.debug(f"Skipping KiteWS auto-check outside market hours (current time: {current_time.strftime('%H:%M:%S IST')})")
            return {
                'checked': False,
                'reason': 'outside_market_hours',
                'message': 'Market is closed, skipping KiteWS check'
            }
        
        # Check if KiteWS process is running
        process_running = False
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'KiteWS.py'],
                capture_output=True,
                text=True
            )
            process_running = result.returncode == 0 and result.stdout.strip()
        except Exception as e:
            logging.debug(f"Could not check KiteWS process status: {e}")
        
        # Check if there's recent data in database (more reliable indicator)
        data_running = check_kitews_running_by_data(minutes_threshold=3)
        
        if process_running and data_running:
            logging.debug("KiteWS is running (both process and data check passed)")
            return {
                'checked': True,
                'running': True,
                'process_running': True,
                'data_running': True,
                'message': 'KiteWS is running normally'
            }
        
        if not data_running:
            # No recent data - KiteWS is not working properly
            logging.warning(f"KiteWS appears to be not running (no recent data in last 3 minutes). Process running: {process_running}")
            
            # If process is running but no data, it might be stuck - we'll still try to start a new one
            # The StartKiteWS wrapper will handle if one is already running
            
            # Auto-start KiteWS
            logging.info("Auto-starting KiteWS due to missing recent data...")
            start_result = start_kitews_automatically()
            
            if start_result:
                return {
                    'checked': True,
                    'running': False,
                    'process_running': process_running,
                    'data_running': False,
                    'auto_started': True,
                    'message': 'KiteWS was not running (no recent data). Auto-started successfully.'
                }
            else:
                return {
                    'checked': True,
                    'running': False,
                    'process_running': process_running,
                    'data_running': False,
                    'auto_started': False,
                    'message': 'KiteWS was not running (no recent data). Auto-start failed.'
                }
        else:
            # Data is running but process check failed - might be a false negative
            logging.info("KiteWS appears to be running (data check passed, but process check inconclusive)")
            return {
                'checked': True,
                'running': True,
                'process_running': process_running,
                'data_running': True,
                'message': 'KiteWS appears to be running (data check passed)'
            }
            
    except Exception as e:
        logging.error(f"Error in check_and_auto_start_kitews: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            'checked': True,
            'running': False,
            'error': str(e),
            'message': f'Error checking/starting KiteWS: {str(e)}'
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
            holdings_info = holdings_service.get_holdings_data(page=page, per_page=10)
            
            # Enrich holdings with today's P&L
            holdings_info = holdings_service.enrich_holdings_with_today_pnl(holdings_info)
            
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "token_status": "Valid" if system_status['token_valid'] else "Invalid",
                "tick_status": "Active" if tick_data['active'] else "Inactive",
                "last_update": system_status['last_update'],
                "total_ticks": tick_data['total_ticks'],
                "holdings_info": holdings_info,
                "show_holdings": SHOW_HOLDINGS,
                "show_stocks_tab": SHOW_STOCKS_TAB,
                "show_options_tab": SHOW_OPTIONS_TAB,
                "show_utilities_tab": SHOW_UTILITIES_TAB,
                "show_fundamentals_tab": SHOW_FUNDAMENTALS_TAB
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
        holdings_info = holdings_service.get_holdings_data(page=1, per_page=10)
        
        # Enrich holdings with today's P&L and Prophet predictions
        holdings_info = holdings_service.enrich_holdings_with_today_pnl(holdings_info)
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "token_status": "Valid" if system_status['token_valid'] else "Invalid",
            "tick_status": "Active" if tick_data['active'] else "Inactive",
            "last_update": system_status['last_update'],
            "total_ticks": tick_data['total_ticks'],
            "holdings_info": holdings_info,
            "show_holdings": SHOW_HOLDINGS,
            "show_stocks_tab": SHOW_STOCKS_TAB,
            "show_options_tab": SHOW_OPTIONS_TAB,
            "show_utilities_tab": SHOW_UTILITIES_TAB,
            "show_fundamentals_tab": SHOW_FUNDAMENTALS_TAB
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

@app.get("/api/cron_jobs_status")
async def api_cron_jobs_status():
    """API endpoint to get status of all cron jobs"""
    try:
        from common.CronJobStatusChecker import CronJobStatusChecker
        
        checker = CronJobStatusChecker()
        status = checker.get_all_jobs_status()
        
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logging.error(f"Error getting cron jobs status: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "status": None
        }

@app.get("/api/cron_job_status/{job_name}")
async def api_cron_job_status(job_name: str):
    """API endpoint to get status of a specific cron job"""
    try:
        from common.CronJobStatusChecker import CronJobStatusChecker
        
        checker = CronJobStatusChecker()
        status = checker.get_job_status(job_name)
        
        if status:
            return {
                "success": True,
                "status": status
            }
        else:
            return {
                "success": False,
                "error": f"Job '{job_name}' not found"
            }
    except Exception as e:
        logging.error(f"Error getting cron job status for {job_name}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "status": None
        }

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
    """API endpoint to generate 5-day TPO chart"""
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
@app.post("/api/start_kitews")
async def api_start_kitews():
    """API endpoint to start KiteWS and run it until 3:30 PM IST"""
    try:
        import subprocess
        import os
        from datetime import datetime
        from pytz import timezone
        
        # Check if KiteWS is already running
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'KiteWS.py'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return {
                    "success": False,
                    "message": "KiteWS is already running",
                    "pid": result.stdout.strip().split('\n')[0]
                }
        except Exception as e:
            # pgrep might not be available, try alternative method
            logging.warning(f"Could not check if KiteWS is running: {e}")
        
        # Start KiteWS using the wrapper script
        kitews_wrapper_path = os.path.join(os.path.dirname(__file__), 'kite', 'StartKiteWS.py')
        
        if not os.path.exists(kitews_wrapper_path):
            return {
                "success": False,
                "error": f"StartKiteWS.py not found at {kitews_wrapper_path}"
            }
        
        # Start the process in the background
        import sys
        process = subprocess.Popen(
            [sys.executable, kitews_wrapper_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(kitews_wrapper_path)),
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        ist = timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        target_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # If current time is past 3:30 PM, set target to next day
        if current_time >= target_time:
            from datetime import timedelta
            target_time = target_time + timedelta(days=1)
        
        logging.info(f"Started KiteWS process (PID: {process.pid}) at {current_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
        logging.info(f"KiteWS will run until {target_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
        
        return {
            "success": True,
            "message": f"KiteWS started successfully. Will run until {target_time.strftime('%Y-%m-%d %H:%M:%S IST')}",
            "pid": process.pid,
            "started_at": current_time.strftime('%Y-%m-%d %H:%M:%S IST'),
            "will_run_until": target_time.strftime('%Y-%m-%d %H:%M:%S IST')
        }
    except Exception as e:
        logging.error(f"Error starting KiteWS: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/api/calculate_tpo")
async def api_calculate_tpo():
    """API endpoint to calculate TPO profile"""
    try:
        import subprocess
        import os
        import sys
        
        # Run CalculateTPO.py as a background process
        calculate_tpo_path = os.path.join(os.path.dirname(__file__), 'market', 'CalculateTPO.py')
        
        if not os.path.exists(calculate_tpo_path):
            return {
                "success": False,
                "error": f"CalculateTPO.py not found at {calculate_tpo_path}"
            }
        
        # Start the process in the background
        process = subprocess.Popen(
            [sys.executable, calculate_tpo_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(calculate_tpo_path),
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        logging.info(f"Started CalculateTPO process (PID: {process.pid})")
        
        return {
            "success": True,
            "message": "TPO calculation started successfully",
            "pid": process.pid
        }
    except Exception as e:
        logging.error(f"Error starting TPO calculation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/api/fetch_options")
async def api_fetch_options():
    """API endpoint to fetch options data from Kite API"""
    try:
        from kite.KiteFetchOptions import fetch_and_save_options
        result = fetch_and_save_options()
        return {
            "success": True,
            "message": "Options data fetched successfully",
            "options_fetched": result.get('options_fetched', 0) if isinstance(result, dict) else 0
        }
    except Exception as e:
        logging.error(f"Error fetching options: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/api/insert_ohlc")
async def api_insert_ohlc():
    """API endpoint to insert OHLC price data"""
    try:
        from kite.InsertOHLC import refresh_stock_prices
        result = refresh_stock_prices()
        return {
            "success": result.get('success', False),
            "message": result.get('message', 'OHLC data updated'),
            "stocks_processed": result.get('stocks_processed', 0),
            "records_inserted": result.get('records_inserted', 0)
        }
    except Exception as e:
        logging.error(f"Error inserting OHLC data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/api/refresh_swing_trades")
async def api_refresh_swing_trades():
    """API endpoint to refresh swing trade recommendations"""
    try:
        from stocks.RefreshSwingTrades import refresh_swing_trades
        refresh_swing_trades()
        return {"success": True, "message": "Swing trade recommendations refreshed successfully"}
    except Exception as e:
        logging.error(f"Error refreshing swing trades: {e}")
        import traceback
        logging.error(traceback.format_exc())
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
                
                # Auto-start KiteWS if not already running (for existing valid tokens)
                try:
                    start_kitews_automatically()
                except Exception as e:
                    logging.debug(f"Could not auto-start KiteWS: {e}")
                
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

@app.get("/api/orderflow_history")
async def api_orderflow_history(
    days: int = Query(5, description="Number of trading days to fetch (default: 5)"),
    instrument_token: int = Query(12683010, description="Instrument token (default: 12683010 for Nifty 50 Futures)")
):
    """API endpoint for historical order flow data"""
    try:
        from market.OrderFlowAnalyzer import OrderFlowAnalyzer
        
        orderflow_analyzer = OrderFlowAnalyzer(instrument_token=instrument_token)
        historical_data = orderflow_analyzer.get_historical_order_flow(days=days)
        
        return historical_data
    except Exception as e:
        logging.error(f"Error fetching historical order flow: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e), "data": []}

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
                    records_inserted += 1 #no of records inserted
                    
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
                    # Get accumulation/distribution data for all swing trade stocks
                    from indicators.AccumulationDistributionAnalyzer import AccumulationDistributionAnalyzer
                    analyzer = AccumulationDistributionAnalyzer()
                    accumulation_map = {}
                    
                    scrip_ids = [row.get('scrip_id') for row in rows if row.get('scrip_id')]
                    for scrip_id in scrip_ids:
                        try:
                            state_data = analyzer.get_current_state(scrip_id, today)
                            if state_data:
                                accumulation_map[scrip_id] = state_data
                        except Exception as e:
                            logging.debug(f"Error getting accumulation/distribution for {scrip_id}: {e}")
                    
                    # Convert database rows to recommendation format
                    recommendations = []
                    for row in rows:
                        scrip_id = row.get('scrip_id')
                        rec = {
                            'scrip_id': scrip_id,
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
                        
                        # Add accumulation/distribution data
                        if scrip_id and scrip_id in accumulation_map:
                            acc_data = accumulation_map[scrip_id]
                            rec['accumulation_state'] = acc_data.get('state')
                            rec['days_in_accumulation_state'] = acc_data.get('days_in_state')
                            rec['accumulation_confidence'] = acc_data.get('confidence_score')
                        else:
                            rec['accumulation_state'] = None
                            rec['days_in_accumulation_state'] = None
                            rec['accumulation_confidence'] = None
                        
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

@app.get("/api/fundamentals/list")
async def api_list_fundamentals(
    scrip_id: Optional[str] = Query(None, description="Specific stock symbol to get (default: all holdings and Nifty50)"),
    include_holdings: bool = Query(True, description="Include fundamentals for all holdings"),
    include_nifty50: bool = Query(True, description="Include fundamentals for Nifty50 stocks"),
    max_days_old: int = Query(30, description="Maximum age of data in days (default: 30)"),
    sort_by: Optional[str] = Query(None, description="Sort by column: sector_code, pe_ratio, roce, market_cap"),
    sort_dir: str = Query("asc", description="Sort direction: asc or desc (default: asc)")
):
    """API endpoint to get existing fundamental data from database (less than max_days_old)"""
    try:
        # Ensure max_days_old is a valid integer
        try:
            max_days_old = int(max_days_old) if max_days_old is not None else 30
            if max_days_old < 0:
                max_days_old = 30
        except (ValueError, TypeError):
            max_days_old = 30
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        results = []
        
        if scrip_id:
            # Get for specific stock - get the latest entry
            cursor.execute("""
                SELECT 
                    fd.scrip_id, fd.fetch_date, fd.pe_ratio, fd.pb_ratio, fd.debt_to_equity,
                    fd.roe, fd.roce, fd.current_ratio, fd.quick_ratio, fd.eps,
                    fd.revenue_growth, fd.profit_growth, fd.dividend_yield, 
                    COALESCE(ms.scrip_mcap, fd.market_cap) as market_cap,
                    ms.scrip_mcap as master_mcap,
                    CURRENT_DATE - fd.fetch_date as days_old,
                    ms.sector_code, ms.scrip_group, ms.yahoo_code
                FROM my_schema.fundamental_data fd
                LEFT JOIN my_schema.master_scrips ms ON ms.scrip_id = fd.scrip_id
                WHERE fd.scrip_id = %s
                AND fd.fetch_date >= CURRENT_DATE - %s::integer
                ORDER BY fd.fetch_date DESC
                LIMIT 1
            """, [scrip_id, max_days_old])
            
            result = cursor.fetchone()
            if result:
                results.append(dict(result))
        else:
            # If both include_holdings and include_nifty50 are True, get ALL fundamental data
            # Otherwise, filter based on the flags
            if include_holdings and include_nifty50:
                # Get ALL fundamental data (most common case)
                cursor.execute("""
                    SELECT DISTINCT ON (fd.scrip_id)
                        fd.scrip_id, fd.fetch_date, fd.pe_ratio, fd.pb_ratio, fd.debt_to_equity,
                        fd.roe, fd.roce, fd.current_ratio, fd.quick_ratio, fd.eps,
                        fd.revenue_growth, fd.profit_growth, fd.dividend_yield, 
                        COALESCE(ms.scrip_mcap, fd.market_cap) as market_cap,
                        ms.scrip_mcap as master_mcap,
                        CURRENT_DATE - fd.fetch_date as days_old,
                        ms.sector_code, ms.scrip_group, ms.yahoo_code
                    FROM my_schema.fundamental_data fd
                    LEFT JOIN my_schema.master_scrips ms ON ms.scrip_id = fd.scrip_id
                    WHERE fd.fetch_date >= CURRENT_DATE - %s::integer
                    ORDER BY fd.scrip_id, fd.fetch_date DESC
                """, [max_days_old])
                
                all_results = cursor.fetchall()
                results.extend([dict(row) for row in all_results])
            else:
                # Get for holdings only
                if include_holdings:
                    cursor.execute("""
                        SELECT DISTINCT ON (fd.scrip_id)
                            fd.scrip_id, fd.fetch_date, fd.pe_ratio, fd.pb_ratio, fd.debt_to_equity,
                            fd.roe, fd.roce, fd.current_ratio, fd.quick_ratio, fd.eps,
                            fd.revenue_growth, fd.profit_growth, fd.dividend_yield, 
                            COALESCE(ms.scrip_mcap, fd.market_cap) as market_cap,
                            ms.scrip_mcap as master_mcap,
                            CURRENT_DATE - fd.fetch_date as days_old,
                            ms.sector_code, ms.scrip_group, ms.yahoo_code
                        FROM my_schema.fundamental_data fd
                        LEFT JOIN my_schema.master_scrips ms ON ms.scrip_id = fd.scrip_id
                        WHERE EXISTS (
                            SELECT 1 FROM my_schema.holdings h 
                            WHERE h.trading_symbol = fd.scrip_id
                        )
                        AND fd.fetch_date >= CURRENT_DATE - %s::integer
                        ORDER BY fd.scrip_id, fd.fetch_date DESC
                    """, [max_days_old])
                    
                    holdings_results = cursor.fetchall()
                    results.extend([dict(row) for row in holdings_results])
                
                # Get for Nifty50 only - get individual constituents (Nifty50 is an index, not a stock)
                if include_nifty50:
                    # Ensure max_days_old is a valid integer for this query
                    try:
                        query_max_days = int(max_days_old) if max_days_old is not None else 30
                        if query_max_days < 0:
                            query_max_days = 30
                    except (ValueError, TypeError):
                        query_max_days = 30
                    
                    # Ensure we have a valid parameter list - use same syntax as other queries
                    # Note: In psycopg2, literal % characters must be escaped as %%
                    cursor.execute("""
                        SELECT DISTINCT ON (fd.scrip_id)
                            fd.scrip_id, fd.fetch_date, fd.pe_ratio, fd.pb_ratio, fd.debt_to_equity,
                            fd.roe, fd.roce, fd.current_ratio, fd.quick_ratio, fd.eps,
                            fd.revenue_growth, fd.profit_growth, fd.dividend_yield, 
                            COALESCE(ms.scrip_mcap, fd.market_cap) as market_cap,
                            ms.scrip_mcap as master_mcap,
                            CURRENT_DATE - fd.fetch_date as days_old,
                            ms.sector_code, ms.scrip_group, ms.yahoo_code
                        FROM my_schema.fundamental_data fd
                        LEFT JOIN my_schema.master_scrips ms ON ms.scrip_id = fd.scrip_id
                        WHERE ms.scrip_id IS NOT NULL
                        AND (ms.scrip_group LIKE '%%NIFTY50%%' OR ms.scrip_group LIKE '%%NIFTY_50%%' OR ms.scrip_group = 'NIFTY50')
                        AND fd.scrip_id NOT IN ('NIFTY50', 'Nifty_50', 'Nifty5')
                        AND fd.fetch_date >= CURRENT_DATE - %s::integer
                        ORDER BY fd.scrip_id, fd.fetch_date DESC
                    """, (query_max_days,))
                    
                    nifty50_results = cursor.fetchall()
                    results.extend([dict(row) for row in nifty50_results])
        
        cursor.close()
        conn.close()
        
        # Apply sorting if requested
        if sort_by and results:
            # Validate sort_by column
            valid_sort_columns = ['sector_code', 'pe_ratio', 'roce', 'market_cap']
            if sort_by.lower() in valid_sort_columns:
                # Validate sort direction
                sort_dir_lower = sort_dir.lower() if sort_dir else 'asc'
                reverse = sort_dir_lower == 'desc'
                
                # Sort the results
                # Separate items with None values from items with valid values
                items_with_values = []
                items_with_none = []
                
                for item in results:
                    value = item.get(sort_by.lower())
                    if value is None:
                        items_with_none.append(item)
                    else:
                        items_with_values.append(item)
                
                # Sort items with valid values
                if items_with_values:
                    def sort_key(item):
                        value = item.get(sort_by.lower())
                        # For numeric values, ensure proper numeric sorting
                        if sort_by.lower() in ['pe_ratio', 'roce', 'market_cap']:
                            try:
                                num_value = float(value) if value is not None else None
                                # Handle None or invalid numeric values - put them at end
                                if num_value is None or (isinstance(num_value, float) and (num_value != num_value)):  # Check for NaN
                                    return float('inf') if not reverse else float('-inf')
                                return num_value
                            except (ValueError, TypeError):
                                return float('inf') if not reverse else float('-inf')
                        # For string values (sector_code)
                        return str(value).lower() if value else ''
                    
                    items_with_values.sort(key=sort_key, reverse=reverse)
                
                # Combine: valid values first, then None values
                results = items_with_values + items_with_none
        
        return {
            "success": True,
            "message": f"Found {len(results)} stocks with fundamental data (less than {max_days_old} days old)",
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logging.error(f"Error listing fundamentals: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }
@app.post("/api/fundamentals/cancel")
async def api_cancel_fundamentals_fetch():
    """API endpoint to cancel ongoing fundamental data batch fetch"""
    try:
        from fundamentals.ScreenerDataFetcher import ScreenerDataFetcher
        
        fetcher = ScreenerDataFetcher()
        fetcher.set_cancellation_flag(True)
        logging.info("Cancellation flag set - user clicked Stop Fetch")
        
        return {
            "success": True,
            "message": "Fundamental data batch fetch cancellation requested"
        }
    except Exception as e:
        logging.error(f"Error cancelling fundamental fetch: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/fundamentals/clear-cancel")
async def api_clear_cancel_flag():
    """API endpoint to clear cancellation flag (called on page load or when fetch starts)"""
    try:
        from fundamentals.ScreenerDataFetcher import ScreenerDataFetcher
        
        fetcher = ScreenerDataFetcher()
        fetcher.set_cancellation_flag(False)
        logging.info("Cancellation flag cleared - page load or fetch started")
        
        return {
            "success": True,
            "message": "Cancellation flag cleared"
        }
    except Exception as e:
        logging.error(f"Error clearing cancellation flag: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/fundamentals/update-mcap")
@app.post("/api/fundamentals/update-mcap")
async def api_update_mcap(
    scrip_id: Optional[str] = Query(None, description="Specific stock symbol to update (default: all holdings and Nifty50)"),
    fetch_holdings: bool = Query(True, description="Update market cap for all holdings"),
    fetch_nifty50: bool = Query(True, description="Update market cap for Nifty50 stocks"),
    batch_size: int = Query(10, description="Number of stocks to process in each batch (default: 10)"),
    delay_between_batches: float = Query(1.0, description="Delay in seconds between batches (default: 1.0)")
):
    """API endpoint to update market cap from Yahoo Finance (monthly refresh)"""
    try:
        from fundamentals.FetchMCap import MarketCapFetcher
        
        fetcher = MarketCapFetcher()
        # Clear cancellation flag at start of new update (user clicked Update)
        fetcher.set_cancellation_flag(False)
        logging.info("Cancellation flag cleared - starting new market cap update")
        
        scrip_ids = [scrip_id] if scrip_id else None
        
        result = fetcher.update_mcap(
            scrip_ids=scrip_ids,
            fetch_holdings=fetch_holdings,
            fetch_nifty50=fetch_nifty50,
            batch_size=batch_size,
            delay_between_batches=delay_between_batches
        )
        
        return {
            "success": result.get('success', True),
            "message": f"Market cap update: {result.get('updated', 0)} updated, {result.get('failed', 0)} failed",
            "summary": {
                "total": result.get('total', 0),
                "updated": result.get('updated', 0),
                "failed": result.get('failed', 0)
            },
            "results": result.get('results', []),
            "note": "Market cap is updated monthly"
        }
    except Exception as e:
        logging.error(f"Error updating market cap: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/fundamentals/cancel-mcap")
async def api_cancel_mcap():
    """API endpoint to cancel market cap update batch processing"""
    try:
        from fundamentals.FetchMCap import MarketCapFetcher
        
        fetcher = MarketCapFetcher()
        fetcher.set_cancellation_flag(True)
        logging.info("Market cap update cancellation requested")
        
        return {
            "success": True,
            "message": "Market cap update batch cancellation requested"
        }
    except Exception as e:
        logging.error(f"Error cancelling market cap update: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/fundamentals/clear-cancel-mcap")
async def api_clear_cancel_mcap():
    """API endpoint to clear cancellation flag for market cap update"""
    try:
        from fundamentals.FetchMCap import MarketCapFetcher
        
        fetcher = MarketCapFetcher()
        fetcher.set_cancellation_flag(False)
        logging.info("Market cap update cancellation flag cleared")
        
        return {
            "success": True,
            "message": "Cancellation flag cleared"
        }
    except Exception as e:
        logging.error(f"Error clearing cancellation flag: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/fundamentals/fetch-yahoo")
@app.post("/api/fundamentals/fetch-yahoo")
async def api_fetch_yahoo_fundamentals(
    scrip_id: Optional[str] = Query(None, description="Specific stock symbol to fetch (default: all holdings and Nifty50)"),
    fetch_holdings: bool = Query(True, description="Fetch Yahoo fundamentals for all holdings"),
    fetch_nifty50: bool = Query(True, description="Fetch Yahoo fundamentals for Nifty50 stocks"),
    batch_size: int = Query(5, description="Number of stocks to process in each batch (default: 5)"),
    delay_between_batches: float = Query(3.45, description="Delay in seconds between batches (default: 3.45)")
):
    """API endpoint to fetch Yahoo Finance fundamentals (actions, income, insider transactions) (monthly refresh)"""
    try:
        from fundamentals.FetchFundamentals import YahooFundamentalsFetcher
        
        fetcher = YahooFundamentalsFetcher()
        # Clear cancellation flag at start of new fetch (user clicked Fetch)
        fetcher.set_cancellation_flag(False)
        logging.info("Cancellation flag cleared - starting new Yahoo fundamentals fetch")
        
        scrip_ids = [scrip_id] if scrip_id else None
        
        result = fetcher.fetch_fundamentals_for_missing_stocks(
            scrip_ids=scrip_ids,
            fetch_holdings=fetch_holdings,
            fetch_nifty50=fetch_nifty50,
            batch_size=batch_size,
            delay_between_batches=delay_between_batches
        )
        
        return {
            "success": result.get('success', True),
            "message": f"Yahoo fundamentals: {result.get('succeeded', 0)} succeeded, {result.get('failed', 0)} failed, {result.get('cancelled', 0)} cancelled",
            "summary": {
                "total": result.get('total', 0),
                "succeeded": result.get('succeeded', 0),
                "failed": result.get('failed', 0),
                "cancelled": result.get('cancelled', 0)
            },
            "results": result.get('results', []),
            "note": "Yahoo fundamentals are fetched monthly"
        }
    except Exception as e:
        logging.error(f"Error fetching Yahoo fundamentals: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/fundamentals/cancel-yahoo")
async def api_cancel_yahoo():
    """API endpoint to cancel Yahoo fundamentals fetch batch processing"""
    try:
        from fundamentals.FetchFundamentals import YahooFundamentalsFetcher
        
        fetcher = YahooFundamentalsFetcher()
        fetcher.set_cancellation_flag(True)
        logging.info("Yahoo fundamentals fetch cancellation requested")
        
        return {
            "success": True,
            "message": "Yahoo fundamentals fetch batch cancellation requested"
        }
    except Exception as e:
        logging.error(f"Error cancelling Yahoo fundamentals fetch: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/fundamentals/clear-cancel-yahoo")
async def api_clear_cancel_yahoo():
    """API endpoint to clear cancellation flag for Yahoo fundamentals fetch"""
    try:
        from fundamentals.FetchFundamentals import YahooFundamentalsFetcher
        
        fetcher = YahooFundamentalsFetcher()
        fetcher.set_cancellation_flag(False)
        logging.info("Yahoo fundamentals fetch cancellation flag cleared")
        
        return {
            "success": True,
            "message": "Cancellation flag cleared"
        }
    except Exception as e:
        logging.error(f"Error clearing cancellation flag: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/fundamentals/fetch")
async def api_fetch_fundamentals(
    scrip_id: Optional[str] = Query(None, description="Specific stock symbol to fetch (default: all holdings and Nifty50)"),
    fetch_holdings: bool = Query(True, description="Fetch fundamentals for all holdings"),
    fetch_nifty50: bool = Query(True, description="Fetch fundamentals for Nifty50 stocks"),
    force_refresh: bool = Query(False, description="Force fetch even if recent data exists (default: only fetch if >30 days old)")
):
    """API endpoint to fetch fundamental data from screener.in (monthly refresh by default)"""
    try:
        from fundamentals.ScreenerDataFetcher import ScreenerDataFetcher
        
        fetcher = ScreenerDataFetcher()
        # Clear cancellation flag at start of new fetch (user clicked Fetch)
        fetcher.set_cancellation_flag(False)
        logging.info("Cancellation flag cleared - starting new fetch")
        results = []
        
        if scrip_id:
            # Fetch for specific stock - check database first
            logging.info(f"Checking database for {scrip_id} before fetching from internet")
            
            # First check if data exists in database
            if fetcher.has_fundamental_data(scrip_id):
                # Data exists, check if we need to refresh (based on age or force_refresh)
                if not fetcher.needs_fundamental_fetch(scrip_id, days_threshold=30, force_refresh=force_refresh):
                    logging.info(f"Skipping {scrip_id} - fundamental data exists in database and is recent (less than 30 days old)")
                    results.append({'scrip_id': scrip_id, 'status': 'skipped', 'reason': 'Data exists in database and is recent'})
                else:
                    logging.info(f"Data exists for {scrip_id} but is old or force_refresh is True, fetching from internet...")
                    data = fetcher.fetch_fundamental_data(scrip_id)
                    if data:
                        fetch_date = date.today()
                        fetcher.save_fundamental_data(scrip_id, data, fetch_date=fetch_date)
                        results.append({'scrip_id': scrip_id, 'status': 'success', 'data': data, 'fetch_date': str(fetch_date)})
                    else:
                        results.append({'scrip_id': scrip_id, 'status': 'failed', 'error': 'No data returned from internet'})
            else:
                logging.info(f"No data in database for {scrip_id}, fetching from internet...")
                data = fetcher.fetch_fundamental_data(scrip_id)
                if data:
                    fetch_date = date.today()
                    fetcher.save_fundamental_data(scrip_id, data, fetch_date=fetch_date)
                    results.append({'scrip_id': scrip_id, 'status': 'success', 'data': data, 'fetch_date': str(fetch_date)})
                else:
                    results.append({'scrip_id': scrip_id, 'status': 'failed', 'error': 'No data returned from internet'})
        else:
            # Fetch for holdings (only if >30 days old, unless forced) - in batches
            if fetch_holdings:
                logging.info("Checking and fetching fundamentals for all holdings (monthly refresh, batch processing)")
                holdings_results = fetcher.fetch_all_holdings_fundamentals(
                    force_refresh=force_refresh,
                    days_threshold=30,
                    batch_size=5,  # Process 5 stocks at a time
                    delay_between_batches=0.0  # No delay between batches
                )
                results.extend(holdings_results)
            
            # Fetch for Nifty50 (only if >30 days old, unless forced) - in batches
            if fetch_nifty50:
                logging.info("Checking and fetching fundamentals for Nifty50 stocks (monthly refresh, batch processing)")
                nifty50_results = fetcher.fetch_nifty50_fundamentals(
                    force_refresh=force_refresh,
                    days_threshold=30,
                    batch_size=5,  # Process 5 stocks at a time
                    delay_between_batches=0.0  # No delay between batches
                )
                results.extend(nifty50_results)
        
        success_count = sum(1 for r in results if r.get('status') == 'success')
        skipped_count = sum(1 for r in results if r.get('status') == 'skipped')
        failed_count = len(results) - success_count - skipped_count
        
        return {
            "success": True,
            "message": f"Fundamentals: {success_count} fetched, {skipped_count} skipped (recent data), {failed_count} failed",
            "results": results,
            "summary": {
                "total": len(results),
                "success": success_count,
                "skipped": skipped_count,
                "failed": failed_count
            },
            "note": "Fundamental data is fetched monthly (only if data is >30 days old) unless force_refresh=true"
        }
    except Exception as e:
        logging.error(f"Error fetching fundamentals: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/sentiment/list_news")
async def api_list_news_sentiment(
    scrip_id: Optional[str] = Query(None, description="Specific stock symbol to get (default: all holdings and Nifty50)"),
    include_holdings: bool = Query(True, description="Include sentiment for all holdings"),
    include_nifty50: bool = Query(True, description="Include sentiment for Nifty50 stocks"),
    max_days_old: int = Query(7, description="Maximum age of data in days (default: 7)")
):
    """API endpoint to get existing news sentiment data from database (recent data)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        results = []
        
        if scrip_id:
            # Get for specific stock - get latest combined sentiment
            cursor.execute("""
                SELECT 
                    cs.scrip_id, cs.calculation_date, 
                    cs.news_sentiment_score, cs.fundamental_sentiment_score, cs.combined_sentiment_score,
                    cs.news_weight, cs.fundamental_weight,
                    COUNT(DISTINCT ns.id) as article_count,
                    CURRENT_DATE - cs.calculation_date as days_old
                FROM my_schema.combined_sentiment cs
                LEFT JOIN my_schema.news_sentiment ns ON ns.scrip_id = cs.scrip_id 
                    AND ns.article_date >= cs.calculation_date - INTERVAL '7 days'
                WHERE cs.scrip_id = %s
                AND CURRENT_DATE - cs.calculation_date <= %s::integer
                GROUP BY cs.scrip_id, cs.calculation_date, cs.news_sentiment_score, 
                         cs.fundamental_sentiment_score, cs.combined_sentiment_score,
                         cs.news_weight, cs.fundamental_weight
                ORDER BY cs.calculation_date DESC
                LIMIT 1
            """, [scrip_id, max_days_old])
            
            result = cursor.fetchone()
            if result:
                results.append(dict(result))
        else:
            # Get for holdings
            if include_holdings:
                cursor.execute("""
                    SELECT DISTINCT ON (cs.scrip_id)
                        cs.scrip_id, cs.calculation_date, 
                        cs.news_sentiment_score, cs.fundamental_sentiment_score, cs.combined_sentiment_score,
                        cs.news_weight, cs.fundamental_weight,
                        COUNT(DISTINCT ns.id) as article_count,
                        CURRENT_DATE - cs.calculation_date as days_old
                    FROM my_schema.combined_sentiment cs
                    JOIN my_schema.holdings h ON h.trading_symbol = cs.scrip_id
                    LEFT JOIN my_schema.news_sentiment ns ON ns.scrip_id = cs.scrip_id 
                        AND ns.article_date >= cs.calculation_date - INTERVAL '7 days'
                    WHERE CURRENT_DATE - cs.calculation_date <= %s::integer
                    GROUP BY cs.scrip_id, cs.calculation_date, cs.news_sentiment_score, 
                             cs.fundamental_sentiment_score, cs.combined_sentiment_score,
                             cs.news_weight, cs.fundamental_weight
                    ORDER BY cs.scrip_id, cs.calculation_date DESC
                """, [max_days_old])
                
                holdings_results = cursor.fetchall()
                results.extend([dict(row) for row in holdings_results])
            
            # Get for Nifty50
            if include_nifty50:
                # Note: In psycopg2, literal % characters must be escaped as %%
                cursor.execute("""
                    SELECT DISTINCT ON (cs.scrip_id)
                        cs.scrip_id, cs.calculation_date, 
                        cs.news_sentiment_score, cs.fundamental_sentiment_score, cs.combined_sentiment_score,
                        cs.news_weight, cs.fundamental_weight,
                        COUNT(DISTINCT ns.id) as article_count,
                        CURRENT_DATE - cs.calculation_date as days_old
                    FROM my_schema.combined_sentiment cs
                    JOIN my_schema.master_scrips ms ON ms.scrip_id = cs.scrip_id
                    LEFT JOIN my_schema.news_sentiment ns ON ns.scrip_id = cs.scrip_id 
                        AND ns.article_date >= cs.calculation_date - INTERVAL '7 days'
                    WHERE (ms.scrip_group LIKE '%%NIFTY50%%' OR ms.scrip_group LIKE '%%NIFTY_50%%' OR ms.scrip_group = 'NIFTY50')
                    AND cs.scrip_id NOT IN ('NIFTY50', 'Nifty_50', 'Nifty5')
                    AND CURRENT_DATE - cs.calculation_date <= %s::integer
                    GROUP BY cs.scrip_id, cs.calculation_date, cs.news_sentiment_score, 
                             cs.fundamental_sentiment_score, cs.combined_sentiment_score,
                             cs.news_weight, cs.fundamental_weight
                    ORDER BY cs.scrip_id, cs.calculation_date DESC
                """, [max_days_old])
                
                nifty50_results = cursor.fetchall()
                results.extend([dict(row) for row in nifty50_results])
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "message": f"Found {len(results)} stocks with news sentiment data (less than {max_days_old} days old)",
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logging.error(f"Error listing news sentiment: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/sentiment/fetch_news")
async def api_fetch_news_sentiment(
    scrip_id: Optional[str] = Query(None, description="Specific stock symbol to analyze (default: all holdings and Nifty50)"),
    days: int = Query(7, description="Number of days to look back for news (default: 7)")
):
    """API endpoint to fetch and analyze news sentiment"""
    try:
        from sentiment.NewsSentimentAnalyzer import NewsSentimentAnalyzer
        
        analyzer = NewsSentimentAnalyzer()
        results = []
        
        if scrip_id:
            # Analyze for specific stock
            logging.info(f"Analyzing news sentiment for {scrip_id}")
            result = analyzer.fetch_and_analyze(scrip_id, days)
            if result:
                results.append(result)
        else:
            # Analyze for all holdings and Nifty50
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get holdings
            cursor.execute("""
                SELECT DISTINCT ms.scrip_id
                FROM my_schema.holdings h
                JOIN my_schema.master_scrips ms ON h.trading_symbol = ms.scrip_id
            """)
            holdings = [row[0] for row in cursor.fetchall()]
            
            # Get Nifty50 stocks
            # Note: In psycopg2, literal % characters must be escaped as %%
            cursor.execute("""
                SELECT DISTINCT scrip_id
                FROM my_schema.master_scrips
                WHERE (scrip_group LIKE '%%NIFTY50%%' OR scrip_group LIKE '%%NIFTY_50%%' OR scrip_group = 'NIFTY50')
                AND scrip_id NOT IN ('NIFTY50', 'Nifty_50', 'Nifty5')
            """)
            nifty50_stocks = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            all_stocks = list(set(holdings + nifty50_stocks))
            
            for stock in all_stocks:
                try:
                    logging.info(f"Analyzing news sentiment for {stock}")
                    result = analyzer.fetch_and_analyze(stock, days)
                    if result:
                        results.append(result)
                except Exception as e:
                    logging.warning(f"Error analyzing news sentiment for {stock}: {e}")
                    continue
        
        return {
            "success": True,
            "message": f"Analyzed news sentiment for {len(results)} stocks",
            "results": results
        }
    except Exception as e:
        logging.error(f"Error fetching news sentiment: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/sentiment/calculate_fundamental")
async def api_calculate_fundamental_sentiment(
    scrip_id: Optional[str] = Query(None, description="Specific stock symbol to analyze (default: all holdings and Nifty50)")
):
    """API endpoint to calculate fundamental sentiment"""
    try:
        from sentiment.FundamentalSentimentAnalyzer import FundamentalSentimentAnalyzer
        from sentiment.CombinedSentimentCalculator import CombinedSentimentCalculator
        
        fundamental_analyzer = FundamentalSentimentAnalyzer()
        combined_calculator = CombinedSentimentCalculator()
        results = []
        
        if scrip_id:
            # Analyze for specific stock
            logging.info(f"Calculating fundamental sentiment for {scrip_id}")
            sentiment_score = fundamental_analyzer.calculate_fundamental_sentiment(scrip_id)
            combined_sentiment = combined_calculator.calculate_combined_sentiment(scrip_id)
            results.append({
                'scrip_id': scrip_id,
                'fundamental_sentiment_score': sentiment_score,
                'combined_sentiment': combined_sentiment
            })
        else:
            # Analyze for all holdings and Nifty50
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get holdings
            cursor.execute("""
                SELECT DISTINCT ms.scrip_id
                FROM my_schema.holdings h
                JOIN my_schema.master_scrips ms ON h.trading_symbol = ms.scrip_id
            """)
            holdings = [row[0] for row in cursor.fetchall()]
            
            # Get Nifty50 stocks
            # Note: In psycopg2, literal % characters must be escaped as %%
            cursor.execute("""
                SELECT DISTINCT scrip_id
                FROM my_schema.master_scrips
                WHERE (scrip_group LIKE '%%NIFTY50%%' OR scrip_group LIKE '%%NIFTY_50%%' OR scrip_group = 'NIFTY50')
                AND scrip_id NOT IN ('NIFTY50', 'Nifty_50', 'Nifty5')
            """)
            nifty50_stocks = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            all_stocks = list(set(holdings + nifty50_stocks))
            
            for stock in all_stocks:
                try:
                    logging.info(f"Calculating fundamental sentiment for {stock}")
                    sentiment_score = fundamental_analyzer.calculate_fundamental_sentiment(stock)
                    combined_sentiment = combined_calculator.calculate_combined_sentiment(stock)
                    results.append({
                        'scrip_id': stock,
                        'fundamental_sentiment_score': sentiment_score,
                        'combined_sentiment': combined_sentiment
                    })
                except Exception as e:
                    logging.warning(f"Error calculating fundamental sentiment for {stock}: {e}")
                    continue
        
        return {
            "success": True,
            "message": f"Calculated fundamental sentiment for {len(results)} stocks",
            "results": results
        }
    except Exception as e:
        logging.error(f"Error calculating fundamental sentiment: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


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
        logging.info(f"[Background] Starting Prophet prediction generation for run_date={run_date}, prediction_days={prediction_days}, limit={limit}")
        predictions = predictor.predict_all_stocks(limit=limit, prediction_days=prediction_days, save_immediately=True, run_date=run_date)
        
        logging.info(f"[Background] Prophet prediction generation completed: {len(predictions)} predictions generated and saved")
        
        if not predictions:
            error_msg = "No predictions generated. This could be due to: insufficient data (need at least 60 days), Prophet model errors, or data quality issues. Check application logs for details."
            logging.warning(f"[Background] {error_msg}")
            return
        
        logging.info(f"[Background] Successfully generated and saved {len(predictions)} predictions for {prediction_days} days")
        
    except Exception as e:
        logging.error(f"[Background] Error generating Prophet predictions: {e}")
        import traceback
        logging.error(traceback.format_exc())


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
        
        logging.info(f"Prophet prediction generation task queued for background execution: run_date={run_date}, prediction_days={prediction_days}, limit={limit}")
        
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
        logging.error(f"Error queuing Prophet prediction generation: {e}")
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
        
        # Calculate days_in_leaderboard and get accumulation/distribution data for each stock
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
                # Note: accumulation/distribution data is already fetched in get_top_gainers() method
                # so we don't need to fetch it again here - it would overwrite the existing data
                for gainer in top_gainers:
                    scrip_id = gainer.get('scrip_id')
                    gainer['days_in_leaderboard'] = days_counts.get(scrip_id, 0)
                    # accumulation_state, days_in_accumulation_state, and accumulation_confidence
                    # are already set by get_top_gainers() method, so we don't overwrite them here
            
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

@app.get("/api/prophet_predictions/top_losers")
async def api_prophet_top_losers(
    limit: int = Query(10, description="Number of top losers to return"),
    prediction_days: int = Query(30, description="Number of prediction days to filter by (default: 30)"),
    force_refresh: bool = Query(False, description="Force refresh (ignore cache)")
):
    """API endpoint to get top N potential losers based on Prophet predictions"""
    try:
        from stocks.ProphetPricePredictor import ProphetPricePredictor
        
        predictor = ProphetPricePredictor()
        top_losers = predictor.get_top_losers(limit=limit, prediction_days=prediction_days)
        
        # Validate that we have actual data
        if not top_losers or len(top_losers) == 0:
            # Check if predictions exist for this prediction_days
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM my_schema.prophet_predictions
                WHERE prediction_days = %s
                AND status = 'ACTIVE'
                AND predicted_price_change_pct < 0
            """, (prediction_days,))
            
            result = cursor.fetchone()
            total_predictions = result[0] if result else 0
            cursor.close()
            conn.close()
            
            if total_predictions == 0:
                return {
                    "success": False,
                    "error": f"No active predictions found for {prediction_days} days with negative price changes",
                    "top_losers": [],
                    "count": 0
                }
            else:
                return {
                    "success": False,
                    "error": f"No top losers found (may need to generate predictions with confidence >= 50%)",
                    "top_losers": [],
                    "count": 0
                }
        
        # Get days_in_leaderboard for each loser
        if top_losers and len(top_losers) > 0:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get scrip_ids from top_losers
            scrip_ids = [loser.get('scrip_id') for loser in top_losers if loser.get('scrip_id')]
            
            if scrip_ids:
                # Get days_in_leaderboard for each scrip_id
                cursor.execute("""
                    SELECT 
                        scrip_id,
                        COUNT(DISTINCT run_date) as days_count
                    FROM my_schema.prophet_predictions
                    WHERE scrip_id = ANY(%s)
                    AND prediction_days = %s
                    AND status = 'ACTIVE'
                    AND predicted_price_change_pct < 0
                    AND prediction_confidence >= 50.0
                    GROUP BY scrip_id
                """, (scrip_ids, prediction_days))
                
                days_counts = {row['scrip_id']: row['days_count'] for row in cursor.fetchall()}
                
                # Add days_in_leaderboard to each loser
                # Note: accumulation/distribution data is already fetched in get_top_losers() method
                # so we don't need to fetch it again here - it would overwrite the existing data
                for loser in top_losers:
                    scrip_id = loser.get('scrip_id')
                    loser['days_in_leaderboard'] = days_counts.get(scrip_id, 0)
                    # accumulation_state, days_in_accumulation_state, and accumulation_confidence
                    # are already set by get_top_losers() method, so we don't overwrite them here
            
            conn.close()
        
        return {
            "success": True,
            "top_losers": top_losers,
            "count": len(top_losers)
        }
        
    except Exception as e:
        logging.error(f"Error getting top losers: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e), "top_losers": []}

@app.get("/api/prophet_predictions/index_projections")
async def api_prophet_index_projections(
    prediction_days: int = Query(60, description="Number of prediction days to filter by (default: 60)"),
    force_refresh: bool = Query(False, description="Force refresh (ignore cache)")
):
    """API endpoint to get Prophet predictions for indices (scrip_group = 'index')"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get indices from master_scrips where scrip_group = 'INDEX' or sector_code = 'INDEX'
        # Using UPPER() for case-insensitive comparison
        cursor.execute("""
            SELECT DISTINCT ms.scrip_id
            FROM my_schema.master_scrips ms
            WHERE UPPER(ms.scrip_group) = 'INDEX' 
            ORDER BY ms.scrip_id
        """)
        indices = cursor.fetchall()
        
        logging.info(f"Found {len(indices)} indices with scrip_group or sector_code = 'INDEX'")
        
        if not indices:
            conn.close()
            return {"success": True, "indices": [], "message": "No indices found in master_scrips"}
        
        # Get predictions for each index
        index_projections = []
        indices_found = [idx['scrip_id'] for idx in indices]
        logging.info(f"Indices found: {indices_found[:10]}...")  # Log first 10
        
        for idx in indices:
            scrip_id = idx['scrip_id']
            
            # First try to get prediction with exact prediction_days (case-insensitive)
            cursor.execute("""
                SELECT 
                    pp.scrip_id,
                    pp.predicted_price_change_pct,
                    pp.prediction_confidence,
                    pp.prediction_days,
                    pp.current_price,
                    pp.predicted_price_30d,
                    pp.run_date
                FROM my_schema.prophet_predictions pp
                WHERE UPPER(pp.scrip_id) = UPPER(%s)
                AND pp.prediction_days = %s
                AND pp.status = 'ACTIVE'
                ORDER BY pp.run_date DESC
                LIMIT 1
            """, (scrip_id, prediction_days))
            
            prediction = cursor.fetchone()
            
            # If no prediction with exact prediction_days, try to get latest prediction regardless of prediction_days
            if not prediction:
                cursor.execute("""
                    SELECT 
                        pp.scrip_id,
                        pp.predicted_price_change_pct,
                        pp.prediction_confidence,
                        pp.prediction_days,
                        pp.current_price,
                        pp.predicted_price_30d,
                        pp.run_date
                    FROM my_schema.prophet_predictions pp
                    WHERE UPPER(pp.scrip_id) = UPPER(%s)
                    AND pp.status = 'ACTIVE'
                    ORDER BY pp.run_date DESC, pp.prediction_days DESC
                    LIMIT 1
                """, (scrip_id,))
                
                prediction = cursor.fetchone()
                
                if prediction:
                    logging.info(f"Found prediction for {scrip_id} with prediction_days={prediction['prediction_days']} (requested {prediction_days})")
                else:
                    logging.debug(f"No prediction found for index {scrip_id}")
            
            if prediction:
                # Get current price if not in prediction
                current_price = prediction['current_price']
                if not current_price:
                    cursor.execute("""
                        SELECT price_close
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        ORDER BY price_date DESC
                        LIMIT 1
                    """, (scrip_id,))
                    price_result = cursor.fetchone()
                    if price_result:
                        current_price = float(price_result['price_close'])
                
                # Calculate predicted price from current price and predicted change percentage
                predicted_price = None
                if current_price and prediction['predicted_price_change_pct']:
                    predicted_price = current_price * (1 + (prediction['predicted_price_change_pct'] / 100))
                
                index_projections.append({
                    "scrip_id": scrip_id,
                    "predicted_price_change_pct": float(prediction['predicted_price_change_pct']) if prediction['predicted_price_change_pct'] else 0.0,
                    "prediction_confidence": float(prediction['prediction_confidence']) if prediction['prediction_confidence'] else 0.0,
                    "prediction_days": int(prediction['prediction_days']) if prediction['prediction_days'] else prediction_days,
                    "current_price": float(current_price) if current_price else None,
                    "predicted_price": float(predicted_price) if predicted_price else None,
                    "run_date": str(prediction['run_date']) if prediction['run_date'] else None
                })
        
        conn.close()
        
        # Sort by predicted price change percentage (descending)
        index_projections.sort(key=lambda x: x['predicted_price_change_pct'], reverse=True)
        
        return {
            "success": True,
            "prediction_days": prediction_days,
            "indices": index_projections
        }
        
    except Exception as e:
        logging.error(f"Error fetching index projections: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e), "indices": []}

@app.get("/api/prophet_predictions/sectoral_averages")
async def api_prophet_sectoral_averages(
    prediction_days: int = Query(30, description="Number of prediction days to filter by (default: 30)"),
    force_refresh: bool = Query(False, description="Force refresh (ignore cache)")
):
    """API endpoint to get sectoral averages for Prophet predictions"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get the latest run_date for the specified prediction_days
        cursor.execute("""
            SELECT MAX(run_date) as max_run_date
            FROM my_schema.prophet_predictions 
            WHERE status = 'ACTIVE' AND prediction_days = %s
        """, (prediction_days,))
        
        result = cursor.fetchone()
        if not result or not result.get('max_run_date'):
            cursor.close()
            conn.close()
            return {
                "success": True,
                "sectoral_averages": [],
                "message": "No predictions found for the specified prediction_days"
            }
        
        run_date = result['max_run_date']
        
        # Get sectoral averages for ALL sectors that have predictions
        # Include all predictions (positive, negative, any confidence) to show complete sectoral view
        # Exclude sectors with 3 or fewer stocks
        cursor.execute("""
            SELECT 
                ms.sector_code,
                AVG(pp.predicted_price_change_pct) as avg_gain_pct,
                AVG(pp.prediction_confidence) as avg_confidence,
                COUNT(pp.scrip_id) as stock_count
            FROM my_schema.prophet_predictions pp
            INNER JOIN my_schema.master_scrips ms ON pp.scrip_id = ms.scrip_id
            WHERE pp.run_date = %s
                AND pp.prediction_days = %s
                AND pp.status = 'ACTIVE'
                AND ms.sector_code IS NOT NULL
                AND ms.sector_code != ''
                AND ms.scrip_country = 'IN'
            GROUP BY ms.sector_code
            HAVING COUNT(pp.scrip_id) > 3
            ORDER BY avg_gain_pct DESC NULLS LAST
        """, (run_date, prediction_days))
        
        rows = cursor.fetchall()
        sectoral_averages = [dict(row) for row in rows]
        
        # Convert numeric types to native Python types
        for sector in sectoral_averages:
            if sector.get('avg_gain_pct') is not None:
                sector['avg_gain_pct'] = float(sector['avg_gain_pct'])
            else:
                sector['avg_gain_pct'] = 0.0
            if sector.get('avg_confidence') is not None:
                sector['avg_confidence'] = float(sector['avg_confidence'])
            else:
                sector['avg_confidence'] = 0.0
            if sector.get('stock_count') is not None:
                sector['stock_count'] = int(sector['stock_count'])
            else:
                sector['stock_count'] = 0
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "sectoral_averages": sectoral_averages,
            "run_date": str(run_date),
            "prediction_days": prediction_days,
            "count": len(sectoral_averages)
        }
        
    except Exception as e:
        logging.error(f"Error getting sectoral averages: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e), "sectoral_averages": []}

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
                
                scrip_id = row.get('scrip_id') or row.get('SCRIP_ID') or row.get('scrip_id')
                gainer_data = {
                    "scrip_id": scrip_id,
                    "gain": float(gain_value) if gain_value is not None else 0.0,
                    "current_price": float(current_price_value) if current_price_value is not None else 0.0,
                    "previous_price": float(previous_price_value) if previous_price_value is not None else 0.0
                }
                gainers_list.append(gainer_data)
            except (ValueError, TypeError, KeyError) as e:
                logging.warning(f"Error processing gainer row for {row.get('scrip_id') or row.get('SCRIP_ID', 'unknown')}: {e}")
                logging.warning(f"Row data: {dict(row)}")
                logging.warning(f"Available keys: {list(row.keys()) if hasattr(row, 'keys') else 'N/A'}")
                continue
        
        # Get accumulation/distribution data for all gainers
        from indicators.AccumulationDistributionAnalyzer import AccumulationDistributionAnalyzer
        from datetime import date
        analyzer = AccumulationDistributionAnalyzer()
        accumulation_map = {}
        
        scrip_ids_for_acc = [g['scrip_id'] for g in gainers_list if g.get('scrip_id')]
        for scrip_id in scrip_ids_for_acc:
            if scrip_id:
                try:
                    state_data = analyzer.get_current_state(scrip_id, date.today())
                    if state_data:
                        accumulation_map[scrip_id] = state_data
                except Exception as e:
                    logging.debug(f"Error getting accumulation/distribution for {scrip_id}: {e}")
        
        # Add accumulation/distribution data to each gainer
        for gainer in gainers_list:
            scrip_id = gainer.get('scrip_id')
            if scrip_id and scrip_id in accumulation_map:
                acc_data = accumulation_map[scrip_id]
                gainer['accumulation_state'] = acc_data.get('state')
                gainer['days_in_accumulation_state'] = acc_data.get('days_in_state')
                gainer['accumulation_confidence'] = acc_data.get('confidence_score')
            else:
                gainer['accumulation_state'] = None
                gainer['days_in_accumulation_state'] = None
                gainer['accumulation_confidence'] = None
        
        # Get Prophet predictions for all gainers (60-day predictions)
        try:
            scrip_ids = [g['scrip_id'] for g in gainers_list if g.get('scrip_id')]
            if scrip_ids:
                # First try to get 60-day predictions
                # Extract cv_mape and cv_rmse from prediction_details JSONB
                # Note: cv_metrics are only available if predictions were generated with PROPHET_ENABLE_CROSS_VALIDATION=true
                cursor.execute("""
                    SELECT scrip_id, 
                           predicted_price_change_pct, 
                           prediction_confidence, 
                           prediction_days,
                           CASE 
                               WHEN prediction_details->'cv_metrics' IS NOT NULL 
                               THEN CAST((prediction_details->'cv_metrics'->>'mape') AS DOUBLE PRECISION)
                               ELSE NULL 
                           END as cv_mape,
                           CASE 
                               WHEN prediction_details->'cv_metrics' IS NOT NULL 
                               THEN CAST((prediction_details->'cv_metrics'->>'rmse') AS DOUBLE PRECISION)
                               ELSE NULL 
                           END as cv_rmse
                    FROM my_schema.prophet_predictions
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE' AND prediction_days = 60)
                    AND prediction_days = 60
                    AND status = 'ACTIVE'
                    AND scrip_id = ANY(%s::text[])
                """, (scrip_ids,))
                
                predictions_rows = cursor.fetchall()
                
                # If no 60-day predictions, try to get latest predictions regardless of prediction_days
                if not predictions_rows:
                    logging.warning("No 60-day Prophet predictions found for gainers, trying latest predictions")
                    cursor.execute("""
                        SELECT scrip_id, 
                               predicted_price_change_pct, 
                               prediction_confidence, 
                               prediction_days,
                               CASE 
                                   WHEN prediction_details->'cv_metrics' IS NOT NULL 
                                   THEN CAST((prediction_details->'cv_metrics'->>'mape') AS DOUBLE PRECISION)
                                   ELSE NULL 
                               END as cv_mape,
                               CASE 
                                   WHEN prediction_details->'cv_metrics' IS NOT NULL 
                                   THEN CAST((prediction_details->'cv_metrics'->>'rmse') AS DOUBLE PRECISION)
                                   ELSE NULL 
                               END as cv_rmse
                        FROM my_schema.prophet_predictions pp1
                        WHERE status = 'ACTIVE'
                        AND run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE')
                        AND scrip_id = ANY(%s::text[])
                    """, (scrip_ids,))
                    predictions_rows = cursor.fetchall()
                
                predictions_map = {}
                for row in predictions_rows:
                    try:
                        scrip_id = row.get('scrip_id') or row.get('SCRIP_ID')
                        if scrip_id:
                            scrip_id_upper = scrip_id.upper()
                            predictions_map[scrip_id_upper] = {
                                'predicted_price_change_pct': row.get('predicted_price_change_pct'),
                                'prediction_confidence': row.get('prediction_confidence'),
                                'prediction_days': row.get('prediction_days'),
                                'cv_mape': row.get('cv_mape'),
                                'cv_rmse': row.get('cv_rmse')
                            }
                            # Debug logging for first prediction
                            if len(predictions_map) == 1:
                                logging.info(f"Sample prediction data: {predictions_map[scrip_id_upper]}")
                    except Exception as e:
                        logging.warning(f"Error processing prediction row: {e}, row: {dict(row)}")
                
                logging.info(f"Loaded {len(predictions_map)} Prophet predictions for gainers enrichment")
                logging.info(f"Scrip IDs in gainers: {[g.get('scrip_id', '').upper() for g in gainers_list]}")
                logging.info(f"Scrip IDs in predictions: {list(predictions_map.keys())}")
                
                # Enrich gainers with Prophet predictions
                for gainer in gainers_list:
                    scrip_id_upper = gainer.get('scrip_id', '').upper()
                    if scrip_id_upper in predictions_map:
                        pred = predictions_map[scrip_id_upper]
                        gainer['prophet_prediction_pct'] = pred.get('predicted_price_change_pct')
                        gainer['prophet_confidence'] = pred.get('prediction_confidence')
                        gainer['prediction_days'] = pred.get('prediction_days', 60)
                        gainer['prophet_cv_mape'] = pred.get('cv_mape')
                        gainer['prophet_cv_rmse'] = pred.get('cv_rmse')
                        logging.debug(f"Enriched gainer {scrip_id_upper}: prediction_pct={gainer['prophet_prediction_pct']}, confidence={gainer['prophet_confidence']}, mape={gainer['prophet_cv_mape']}, rmse={gainer['prophet_cv_rmse']}")
                    else:
                        gainer['prophet_prediction_pct'] = None
                        gainer['prophet_confidence'] = None
                        gainer['prediction_days'] = None
                        gainer['prophet_cv_mape'] = None
                        gainer['prophet_cv_rmse'] = None
                        logging.debug(f"No prediction found for gainer {scrip_id_upper}")
        except Exception as e:
            logging.error(f"Error loading Prophet predictions for gainers: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Set None for all gainers if error occurs
            for gainer in gainers_list:
                gainer['prophet_prediction_pct'] = None
                gainer['prophet_confidence'] = None
                gainer['prediction_days'] = None
                gainer['prophet_cv_mape'] = None
                gainer['prophet_cv_rmse'] = None
        
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
        
        # Get accumulation/distribution data for all losers
        from indicators.AccumulationDistributionAnalyzer import AccumulationDistributionAnalyzer
        from datetime import date
        analyzer = AccumulationDistributionAnalyzer()
        accumulation_map = {}
        
        scrip_ids_for_acc = [l['scrip_id'] for l in losers_list if l.get('scrip_id')]
        for scrip_id in scrip_ids_for_acc:
            if scrip_id:
                try:
                    state_data = analyzer.get_current_state(scrip_id, date.today())
                    if state_data:
                        accumulation_map[scrip_id] = state_data
                except Exception as e:
                    logging.debug(f"Error getting accumulation/distribution for {scrip_id}: {e}")
        
        # Add accumulation/distribution data to each loser
        for loser in losers_list:
            scrip_id = loser.get('scrip_id')
            if scrip_id and scrip_id in accumulation_map:
                acc_data = accumulation_map[scrip_id]
                loser['accumulation_state'] = acc_data.get('state')
                loser['days_in_accumulation_state'] = acc_data.get('days_in_state')
                loser['accumulation_confidence'] = acc_data.get('confidence_score')
            else:
                loser['accumulation_state'] = None
                loser['days_in_accumulation_state'] = None
                loser['accumulation_confidence'] = None
        
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
@app.get("/api/accumulation_distribution/{scrip_id}")
async def api_accumulation_distribution(scrip_id: str, analysis_date: str = Query(None)):
    """API endpoint to get accumulation/distribution state for a stock"""
    try:
        from indicators.AccumulationDistributionAnalyzer import AccumulationDistributionAnalyzer
        from datetime import date, datetime
        
        analyzer = AccumulationDistributionAnalyzer()
        
        # Parse analysis_date or use today
        if analysis_date:
            try:
                analysis_date_obj = datetime.strptime(analysis_date, '%Y-%m-%d').date()
            except ValueError:
                analysis_date_obj = date.today()
        else:
            analysis_date_obj = date.today()
        
        # Get current state from database
        state_data = analyzer.get_current_state(scrip_id, analysis_date_obj)
        
        if state_data:
            return cached_json_response({
                "success": True,
                "scrip_id": scrip_id,
                "analysis_date": analysis_date_obj.isoformat(),
                "state": state_data.get('state'),
                "start_date": state_data.get('start_date').isoformat() if state_data.get('start_date') else None,
                "days_in_state": state_data.get('days_in_state'),
                "obv_value": state_data.get('obv_value'),
                "ad_value": state_data.get('ad_value'),
                "momentum_score": state_data.get('momentum_score'),
                "pattern_detected": state_data.get('pattern_detected'),
                "confidence_score": state_data.get('confidence_score'),
                "volume_analysis": state_data.get('volume_analysis'),
                "technical_context": state_data.get('technical_context')
            }, f"/api/accumulation_distribution/{scrip_id}")
        else:
            return cached_json_response({
                "success": False,
                "error": f"No accumulation/distribution data found for {scrip_id} on {analysis_date_obj}"
            }, f"/api/accumulation_distribution/{scrip_id}")
            
    except Exception as e:
        logging.error(f"Error getting accumulation/distribution for {scrip_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


@app.get("/api/accumulation_distribution/batch")
async def api_accumulation_distribution_batch(scrip_ids: str = Query(None), analysis_date: str = Query(None)):
    """API endpoint to get accumulation/distribution states for multiple stocks"""
    try:
        from indicators.AccumulationDistributionAnalyzer import AccumulationDistributionAnalyzer
        from datetime import date, datetime
        import json
        
        analyzer = AccumulationDistributionAnalyzer()
        
        # Parse scrip_ids (comma-separated)
        if scrip_ids:
            scrip_id_list = [s.strip() for s in scrip_ids.split(',')]
        else:
            return {"success": False, "error": "scrip_ids parameter required"}
        
        # Parse analysis_date or use today
        if analysis_date:
            try:
                analysis_date_obj = datetime.strptime(analysis_date, '%Y-%m-%d').date()
            except ValueError:
                analysis_date_obj = date.today()
        else:
            analysis_date_obj = date.today()
        
        # Get states for all stocks
        results = {}
        for scrip_id in scrip_id_list:
            state_data = analyzer.get_current_state(scrip_id, analysis_date_obj)
            if state_data:
                results[scrip_id] = {
                    "state": state_data.get('state'),
                    "start_date": state_data.get('start_date').isoformat() if state_data.get('start_date') else None,
                    "days_in_state": state_data.get('days_in_state'),
                    "obv_value": state_data.get('obv_value'),
                    "ad_value": state_data.get('ad_value'),
                    "momentum_score": state_data.get('momentum_score'),
                    "pattern_detected": state_data.get('pattern_detected'),
                    "confidence_score": state_data.get('confidence_score')
                }
            else:
                results[scrip_id] = None
        
        return cached_json_response({
            "success": True,
            "analysis_date": analysis_date_obj.isoformat(),
            "results": results
        }, f"/api/accumulation_distribution/batch?scrip_ids={scrip_ids}")
        
    except Exception as e:
        logging.error(f"Error getting batch accumulation/distribution: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


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

@app.get("/api/candlestick/{trading_symbol}")
async def api_candlestick(trading_symbol: str, days: int = Query(30, ge=7, le=90), prediction_days: int = Query(None, description="Number of prediction days to filter by (optional)")):
    """API endpoint to get candlestick chart data for a stock"""
    try:
        logging.info(f"Candlestick API called for trading_symbol='{trading_symbol}' (type: {type(trading_symbol)}, len: {len(trading_symbol) if trading_symbol else 0})")
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
        volumes_array = np.array(volumes)
        
        # Calculate OBV (On Balance Volume)
        def calculate_obv(close, volume):
            """Calculate On Balance Volume indicator"""
            obv = np.zeros(len(close))
            if len(close) > 0 and len(volume) > 0:
                obv[0] = volume[0]
                for i in range(1, len(close)):
                    if close[i] > close[i-1]:
                        obv[i] = obv[i-1] + volume[i]
                    elif close[i] < close[i-1]:
                        obv[i] = obv[i-1] - volume[i]
                    else:
                        obv[i] = obv[i-1]
            return obv
        
        obv = calculate_obv(closes_array, volumes_array)
        
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
                "obv": round(float(obv[i]), 2) if not np.isnan(obv[i]) else None,
                "sma_20": round(float(sma_20[i]), 2) if not np.isnan(sma_20[i]) else None,
                "sma_50": round(float(sma_50[i]), 2) if not np.isnan(sma_50[i]) else None,
                "sma_200": round(float(sma_200[i]), 2) if not np.isnan(sma_200[i]) else None,
                "supertrend": round(float(supertrend[i]), 2) if not np.isnan(supertrend[i]) else None,
                "supertrend_direction": int(supertrend_direction[i]) if not np.isnan(supertrend_direction[i]) else None
            })
        
        # Get fundamental data (P/E, ROCE, Market Cap)
        fundamental_data = {}
        try:
            cursor.execute("""
                SELECT 
                    pe_ratio,
                    roce,
                    market_cap
                FROM my_schema.fundamental_data fd
                WHERE fd.scrip_id = %s
                ORDER BY fd.fetch_date DESC
                LIMIT 1
            """, (trading_symbol,))
            fundamental_result = cursor.fetchone()
            if fundamental_result:
                fundamental_data = {
                    "pe_ratio": float(fundamental_result['pe_ratio']) if fundamental_result['pe_ratio'] else None,
                    "roce": float(fundamental_result['roce']) if fundamental_result['roce'] else None,
                    "market_cap": float(fundamental_result['market_cap']) if fundamental_result['market_cap'] else None
                }
        except Exception as e:
            logging.debug(f"Could not fetch fundamental data for {trading_symbol}: {e}")
        
        # Get Prophet prediction data (ENHANCED: add Ghost Prediction)
        prediction_data = None
        try:
            # Build query with optional prediction_days filter
            if prediction_days is not None:
                # First try with ACTIVE status and specific prediction_days
                cursor.execute("""
                    SELECT 
                        predicted_price_change_pct,
                        prediction_confidence,
                        prediction_days,
                        prediction_details,
                        run_date,
                        status
                    FROM my_schema.prophet_predictions
                    WHERE UPPER(TRIM(scrip_id)) = UPPER(TRIM(%s))
                    AND (status = 'ACTIVE' OR status IS NULL)
                    AND prediction_days = %s
                    ORDER BY run_date DESC
                    LIMIT 1
                """, (trading_symbol, prediction_days))
                prediction_result = cursor.fetchone()
                logging.info(f"Prediction query (ACTIVE, prediction_days={prediction_days}) for {trading_symbol}: found={prediction_result is not None}")
                
                # If not found, try without status filter but with prediction_days
                if not prediction_result:
                    cursor.execute("""
                        SELECT 
                            predicted_price_change_pct,
                            prediction_confidence,
                            prediction_days,
                            prediction_details,
                            run_date,
                            status
                        FROM my_schema.prophet_predictions
                        WHERE UPPER(TRIM(scrip_id)) = UPPER(TRIM(%s))
                        AND prediction_days = %s
                        ORDER BY run_date DESC
                        LIMIT 1
                    """, (trading_symbol, prediction_days))
                    prediction_result = cursor.fetchone()
                    logging.info(f"Prediction query (any status, prediction_days={prediction_days}) for {trading_symbol}: found={prediction_result is not None}")
            else:
                # First try with ACTIVE status (original behavior)
                cursor.execute("""
                    SELECT 
                        predicted_price_change_pct,
                        prediction_confidence,
                        prediction_days,
                        prediction_details,
                        run_date,
                        status
                    FROM my_schema.prophet_predictions
                    WHERE UPPER(TRIM(scrip_id)) = UPPER(TRIM(%s))
                    AND (status = 'ACTIVE' OR status IS NULL)
                    ORDER BY run_date DESC, prediction_days DESC
                    LIMIT 1
                """, (trading_symbol,))
                prediction_result = cursor.fetchone()
                logging.info(f"Prediction query (ACTIVE) for {trading_symbol}: found={prediction_result is not None}")
                
                # If not found, try without status filter
                if not prediction_result:
                    cursor.execute("""
                        SELECT 
                            predicted_price_change_pct,
                            prediction_confidence,
                            prediction_days,
                            prediction_details,
                            run_date,
                            status
                        FROM my_schema.prophet_predictions
                        WHERE UPPER(TRIM(scrip_id)) = UPPER(TRIM(%s))
                        ORDER BY run_date DESC, prediction_days DESC
                        LIMIT 1
                    """, (trading_symbol,))
                    prediction_result = cursor.fetchone()
                    logging.info(f"Prediction query (any status) for {trading_symbol}: found={prediction_result is not None}")
            
            # Debug: Check what scrip_ids exist in the table
            if not prediction_result:
                cursor.execute("""
                    SELECT DISTINCT scrip_id, status, COUNT(*) as cnt
                    FROM my_schema.prophet_predictions
                    WHERE scrip_id ILIKE %s
                    GROUP BY scrip_id, status
                    LIMIT 5
                """, (f'%{trading_symbol}%',))
                similar_results = cursor.fetchall()
                if similar_results:
                    logging.info(f"Similar scrip_ids found for {trading_symbol}: {[dict(r) for r in similar_results]}")
            
            if prediction_result:
                logging.info(f"Prediction data for {trading_symbol}: change_pct={prediction_result.get('predicted_price_change_pct')}, confidence={prediction_result.get('prediction_confidence')}, days={prediction_result.get('prediction_days')}")
                # Get latest close price to calculate predicted prices
                latest_close = closes[-1] if len(closes) > 0 else None
                predicted_price = None
                if latest_close and prediction_result['predicted_price_change_pct']:
                    predicted_price = latest_close * (1 + (prediction_result['predicted_price_change_pct'] / 100))
                
                # Parse prediction_details if available (contains daily forecasts)
                daily_predictions = None
                if prediction_result['prediction_details']:
                    try:
                        import json
                        if isinstance(prediction_result['prediction_details'], str):
                            details = json.loads(prediction_result['prediction_details'])
                        else:
                            details = prediction_result['prediction_details']
                        
                        # Extract daily predictions if available
                        if isinstance(details, dict) and 'daily_forecasts' in details:
                            daily_predictions = details['daily_forecasts']
                    except Exception as e:
                        logging.debug(f"Could not parse prediction_details for {trading_symbol}: {e}")
                
                prediction_data = {
                    "predicted_price_change_pct": float(prediction_result['predicted_price_change_pct']) if prediction_result['predicted_price_change_pct'] else None,
                    "prediction_confidence": float(prediction_result['prediction_confidence']) if prediction_result['prediction_confidence'] else None,
                    "prediction_days": int(prediction_result['prediction_days']) if prediction_result['prediction_days'] else None,
                    "predicted_price": float(predicted_price) if predicted_price else None,
                    "current_price": float(latest_close) if latest_close else None,
                    "daily_predictions": daily_predictions,
                    "run_date": str(prediction_result['run_date']) if prediction_result['run_date'] else None
                }
                logging.info(f"Prediction data created for {trading_symbol}: predicted_price={prediction_data.get('predicted_price')}, current_price={prediction_data.get('current_price')}")
            else:
                logging.debug(f"No prediction found for {trading_symbol} with status='ACTIVE'")
        except Exception as e:
            logging.error(f"Error fetching prediction data for {trading_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        # Get Accumulation/Distribution state (ENHANCED)
        accumulation_data = None
        try:
            from datetime import date as date_class
            today = date_class.today()
            # Try with exact match first (case-insensitive)
            cursor.execute("""
                SELECT 
                    state,
                    start_date,
                    days_in_state,
                    confidence_score,
                    pattern_detected,
                    technical_context,
                    analysis_date
                FROM my_schema.accumulation_distribution
                WHERE UPPER(TRIM(scrip_id)) = UPPER(TRIM(%s))
                AND analysis_date = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (trading_symbol, today))
            accumulation_result = cursor.fetchone()
            logging.info(f"Accumulation query (today) for {trading_symbol}: found={accumulation_result is not None}")
            
            # Debug: Check what scrip_ids exist in the table
            if not accumulation_result:
                cursor.execute("""
                    SELECT DISTINCT scrip_id, analysis_date, state
                    FROM my_schema.accumulation_distribution
                    WHERE scrip_id ILIKE %s
                    ORDER BY analysis_date DESC
                    LIMIT 5
                """, (f'%{trading_symbol}%',))
                similar_results = cursor.fetchall()
                if similar_results:
                    logging.info(f"Similar scrip_ids found for {trading_symbol}: {[dict(r) for r in similar_results]}")
            
            if accumulation_result:
                accumulation_data = {
                    "state": accumulation_result['state'] if accumulation_result['state'] else None,
                    "start_date": str(accumulation_result['start_date']) if accumulation_result['start_date'] else None,
                    "days_in_state": int(accumulation_result['days_in_state']) if accumulation_result['days_in_state'] else None,
                    "confidence_score": float(accumulation_result['confidence_score']) if accumulation_result['confidence_score'] else None,
                    "pattern_detected": accumulation_result['pattern_detected'] if accumulation_result['pattern_detected'] else None,
                    "technical_context": accumulation_result['technical_context'] if accumulation_result['technical_context'] else None
                }
                logging.info(f"Accumulation data for {trading_symbol}: state={accumulation_data.get('state')}, confidence={accumulation_data.get('confidence_score')}")
            else:
                # Try to get the latest available state if today's data is not available
                cursor.execute("""
                    SELECT 
                        state,
                        start_date,
                        days_in_state,
                        confidence_score,
                        pattern_detected,
                        technical_context,
                        analysis_date
                    FROM my_schema.accumulation_distribution
                    WHERE UPPER(TRIM(scrip_id)) = UPPER(TRIM(%s))
                    ORDER BY analysis_date DESC
                    LIMIT 1
                """, (trading_symbol,))
                accumulation_result = cursor.fetchone()
                logging.info(f"Accumulation query (latest) for {trading_symbol}: found={accumulation_result is not None}")
                if accumulation_result:
                    accumulation_data = {
                        "state": accumulation_result['state'] if accumulation_result['state'] else None,
                        "start_date": str(accumulation_result['start_date']) if accumulation_result['start_date'] else None,
                        "days_in_state": int(accumulation_result['days_in_state']) if accumulation_result['days_in_state'] else None,
                        "confidence_score": float(accumulation_result['confidence_score']) if accumulation_result['confidence_score'] else None,
                        "pattern_detected": accumulation_result['pattern_detected'] if accumulation_result['pattern_detected'] else None,
                        "technical_context": accumulation_result['technical_context'] if accumulation_result['technical_context'] else None
                    }
                    logging.info(f"Accumulation data (latest) for {trading_symbol}: state={accumulation_data.get('state')}")
        except Exception as e:
            logging.debug(f"Could not fetch accumulation/distribution data for {trading_symbol}: {e}")
        
        conn.close()
        
        return {
            "trading_symbol": trading_symbol,
            "data": data,
            "fundamental_data": fundamental_data,
            "prediction_data": prediction_data,
            "accumulation_data": accumulation_data
        }
    except Exception as e:
        logging.error(f"Error fetching candlestick data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e), "data": []}

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
        
        # Get filtering statistics
        filtering_stats = getattr(suggestion_engine, '_last_filtering_stats', {
            'initial_count': len(suggestions),
            'filtered_exhaustion': 0,
            'filtered_sentiment_conflict': 0,
            'filtered_pressure_conflict': 0,
            'filtered_weak_confidence': 0,
            'final_count': len(suggestions),
            'futures_generated': 0,
            'options_generated': 0,
            'generation_issue': len(suggestions) == 0
        })
        
        return {
            "success": True,
            "analysis_date": analysis_date or datetime.now().strftime('%Y-%m-%d'),
            "instrument_token": instrument_token,
            "current_price": current_price,
            "suggestions": suggestions,
            "total_suggestions": len(suggestions),
            "filtering_stats": filtering_stats
        }
    except Exception as e:
        logging.error(f"Error generating derivatives suggestions: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "suggestions": [],
            "filtering_stats": {
                'initial_count': 0,
                'filtered_exhaustion': 0,
                'filtered_sentiment_conflict': 0,
                'filtered_pressure_conflict': 0,
                'filtered_weak_confidence': 0,
                'final_count': 0,
                'futures_generated': 0,
                'options_generated': 0,
                'generation_issue': True
            }
        }

@app.get("/api/intraday_options_suggestions")
async def api_intraday_options_suggestions(
    instrument_token: int = Query(256265, description="Instrument token (default: 256265 for Nifty 50)"),
    analysis_date: str = Query(None, description="Analysis date in YYYY-MM-DD format (default: today)"),
    timeframe_minutes: int = Query(15, description="Timeframe in minutes (default: 15, max: 240)"),
    timeframe_days: int = Query(None, description="Timeframe in days (1, 3, or 5). If provided, this takes precedence over timeframe_minutes")
):
    """
    API endpoint to get options trading suggestions based on:
    - Intraday POC changes (for short-term)
    - Daily POC trend analysis (for long-term)
    - Futures and Options order flow
    - Supports both minutes (5-240) and days (1, 3, 5) timeframes
    """
    try:
        from options.IntradayOptionsSuggestions import IntradayOptionsSuggestions
        from datetime import date
        
        # Parse analysis date
        if analysis_date:
            try:
                analysis_date_obj = datetime.strptime(analysis_date, '%Y-%m-%d').date()
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid date format. Use YYYY-MM-DD"
                }
        else:
            analysis_date_obj = date.today()
        
        # Validate timeframe
        if timeframe_days is not None:
            # Days timeframe
            if timeframe_days not in [1, 3, 5]:
                return {
                    "success": False,
                    "error": "Timeframe days must be 1, 3, or 5"
                }
            # Use default minutes for intraday POC calculation, but generate long-term suggestions
            timeframe_minutes_for_calc = 15
        else:
            # Minutes timeframe
            if timeframe_minutes < 5 or timeframe_minutes > 240:
                return {
                    "success": False,
                    "error": "Timeframe minutes must be between 5 and 240 minutes"
                }
            timeframe_minutes_for_calc = timeframe_minutes
        
        # Get database configuration
        db_config = {
            'host': os.getenv('PG_HOST', 'postgres'),
            'database': os.getenv('PG_DATABASE', 'mydb'),
            'user': os.getenv('PG_USER', 'postgres'),
            'password': os.getenv('PG_PASSWORD', 'postgres'),
            'port': int(os.getenv('PG_PORT', 5432))
        }
        
        # Get active futures token dynamically from database
        # Futures contracts expire monthly, so we need to fetch the active contract
        futures_token = None
        try:
            from common.Boilerplate import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Try to find the active futures contract token for today
            cursor.execute("""
                SELECT instrument_token
                FROM (
                    SELECT instrument_token, timestamp
                    FROM my_schema.futures_ticks
                    WHERE run_date = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                ) AS subquery
            """, (analysis_date_obj,))
            futures_result = cursor.fetchone()
            if futures_result:
                futures_token = futures_result[0]
                logging.info(f"Found active futures contract token: {futures_token} for date {analysis_date_obj}")
            else:
                # Fallback to default if no data found
                futures_token = 12683010
                logging.warning(f"No futures data found for {analysis_date_obj}, using default token {futures_token}")
            
            conn.close()
        except Exception as e:
            logging.warning(f"Could not fetch active futures token: {e}, using default 12683010")
            futures_token = 12683010  # Fallback to default
        
        # Initialize options suggestions engine
        suggestions_engine = IntradayOptionsSuggestions(
            instrument_token=instrument_token,
            futures_token=futures_token,  # Dynamically fetched active futures token
            tick_size=5.0,
            timeframe_minutes=timeframe_minutes_for_calc,
            db_config=db_config
        )
        
        # Generate suggestions
        result = suggestions_engine.generate_suggestions(
            analysis_date=analysis_date_obj,
            current_time=None  # Will use current time
        )
        
        # If timeframe_days is specified, regenerate long-term suggestions with specific days
        if timeframe_days is not None and result.get('success'):
            # Get the options chain and other data needed for long-term suggestions
            current_price = result.get('current_price', 0)
            current_poc = result.get('current_poc', 0)
            poc_change = result.get('poc_change', {})
            order_flow = {'overall_sentiment': result.get('order_flow_sentiment', 'Neutral')}
            
            # Get options chain (we need to fetch it again or use the one from the engine)
            options_chain = suggestions_engine._get_relevant_options(current_price, current_poc)
            
            # Regenerate long-term suggestions with the specific timeframe_days
            long_term_suggestions = suggestions_engine._generate_long_term_suggestions(
                current_price=current_price,
                current_poc=current_poc,
                poc_change=poc_change,
                order_flow=order_flow,
                options_chain=options_chain,
                analysis_date=analysis_date_obj,
                timeframe_days=timeframe_days
            )
            
            result['long_term_suggestions'] = long_term_suggestions
            result['short_term_suggestions'] = []  # Hide short-term when days are selected
            result['timeframe_days'] = timeframe_days
        
        return result
        
    except Exception as e:
        logging.error(f"Error generating intraday options suggestions: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "suggestions": []
        }

@app.post("/api/options_payoff_chart")
async def api_options_payoff_chart(body: dict):
    """
    API endpoint to generate payoff chart with Greeks for an options strategy
    
    Request body should contain:
    - strategy_legs: List of strategy legs, each with:
        - action: 'BUY' or 'SELL'
        - quantity: Number of lots
        - option_type: 'CE' or 'PE'
        - strike: Strike price
        - premium: Premium paid/received
        - expiry: Expiry date (YYYY-MM-DD)
    - current_price: Current underlying price
    - expiry_date: Expiry date (YYYY-MM-DD)
    - implied_volatility: Implied volatility as percentage (default: 20)
    - lot_size: Lot size (default: 50)
    """
    try:
        from options.OptionsStrategies import OptionsStrategies
        from datetime import date
        
        strategy_legs = body.get('strategy_legs', [])
        current_price = float(body.get('current_price', 0))
        expiry_date_str = body.get('expiry_date')
        implied_volatility_pct = float(body.get('implied_volatility', 20))
        lot_size = int(body.get('lot_size', 50))
        
        if not strategy_legs:
            return {
                "success": False,
                "error": "strategy_legs is required"
            }
        
        if current_price <= 0:
            return {
                "success": False,
                "error": "current_price must be greater than 0"
            }
        
        # Parse expiry date
        try:
            expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            # Try to get expiry from first leg
            if strategy_legs and 'expiry' in strategy_legs[0]:
                try:
                    expiry_date = datetime.strptime(strategy_legs[0]['expiry'], '%Y-%m-%d').date()
                except:
                    return {
                        "success": False,
                        "error": "Invalid expiry_date format. Use YYYY-MM-DD"
                    }
            else:
                return {
                    "success": False,
                    "error": "expiry_date is required"
                }
        
        # Convert IV from percentage to decimal
        implied_volatility = implied_volatility_pct / 100.0
        
        # Initialize strategies generator
        strategies_generator = OptionsStrategies(risk_free_rate=0.065)
        
        # Generate payoff chart with Greeks
        result = strategies_generator.generate_payoff_chart_with_greeks(
            strategy_legs=strategy_legs,
            current_price=current_price,
            expiry_date=expiry_date,
            implied_volatility=implied_volatility,
            lot_size=lot_size,
            current_date=date.today()
        )
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        logging.error(f"Error generating payoff chart: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
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

@app.get("/api/options_oi_history")
async def api_options_oi_history(
    tradingsymbol: str = Query(..., description="Trading symbol of the option"),
    strike_price: float = Query(..., description="Strike price"),
    option_type: str = Query(..., description="Option type: 'CE' or 'PE'"),
    expiry: str = Query(None, description="Expiry date in YYYY-MM-DD format"),
    days: int = Query(7, description="Number of days of history to fetch (default: 7)")
):
    """
    API endpoint to get historical OI data for a specific option
    Returns OI values over time for the specified trading symbol and strike
    """
    try:
        from datetime import datetime, date, timedelta
        from common.Boilerplate import get_db_connection
        
        # Parse expiry date
        expiry_date = None
        if expiry:
            try:
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid expiry date format. Use YYYY-MM-DD"
                }
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query to get historical OI data for this specific option
        where_conditions = [
            "tradingsymbol = %s",
            "strike_price = %s",
            "option_type = %s",
            "run_date >= %s",
            "run_date <= %s"
        ]
        params = [tradingsymbol, strike_price, option_type, start_date, end_date]
        
        if expiry_date:
            where_conditions.append("expiry = %s")
            params.append(expiry_date)
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
            SELECT 
                run_date,
                timestamp,
                oi,
                oi_day_high,
                oi_day_low,
                volume,
                last_price
            FROM my_schema.options_ticks
            WHERE {where_clause}
            ORDER BY run_date ASC, timestamp ASC
        """
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                "success": False,
                "error": "No historical OI data found for this option",
                "oi_history": []
            }
        
        # Convert to list of dicts
        oi_history = []
        for row in rows:
            oi_history.append({
                "run_date": row[0].isoformat() if isinstance(row[0], date) else str(row[0]),
                "timestamp": row[1].isoformat() if hasattr(row[1], 'isoformat') else str(row[1]),
                "oi": int(row[2]) if row[2] else 0,
                "oi_day_high": int(row[3]) if row[3] else 0,
                "oi_day_low": int(row[4]) if row[4] else 0,
                "volume": int(row[5]) if row[5] else 0,
                "last_price": float(row[6]) if row[6] else 0
            })
        
        return {
            "success": True,
            "tradingsymbol": tradingsymbol,
            "strike_price": strike_price,
            "option_type": option_type,
            "expiry": expiry_date.isoformat() if expiry_date else None,
            "oi_history": oi_history,
            "total_records": len(oi_history)
        }
        
    except Exception as e:
        logging.error(f"Error fetching OI history: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "oi_history": []
        }

@app.get("/api/options_oi_chart_for_strike")
async def api_options_oi_chart_for_strike(
    tradingsymbol: str = Query(..., description="Trading symbol of the option"),
    strike_price: float = Query(..., description="Strike price"),
    option_type: str = Query(..., description="Option type: 'CE' or 'PE'"),
    expiry: str = Query(None, description="Expiry date in YYYY-MM-DD format"),
    days: int = Query(7, description="Number of days of history to show (default: 7)")
):
    """
    API endpoint to generate OI change over time chart for a specific strike price
    """
    try:
        from options.OptionsOIAnalyzer import OptionsOIAnalyzer
        from datetime import datetime, date
        
        analyzer = OptionsOIAnalyzer()
        
        # Parse expiry date
        expiry_date = None
        if expiry:
            try:
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid expiry date format. Use YYYY-MM-DD"
                }
        
        # Validate option_type
        if option_type not in ['CE', 'PE']:
            return {
                "success": False,
                "error": "option_type must be 'CE' or 'PE'"
            }
        
        # Generate chart
        chart_image = analyzer.plot_oi_history_for_strike(
            tradingsymbol=tradingsymbol,
            strike_price=strike_price,
            option_type=option_type,
            expiry=expiry_date,
            days=days
        )
        
        if not chart_image:
            return {
                "success": False,
                "error": "Failed to generate chart"
            }
        
        return {
            "success": True,
            "tradingsymbol": tradingsymbol,
            "strike_price": strike_price,
            "option_type": option_type,
            "expiry": expiry_date.isoformat() if expiry_date else None,
            "days": days,
            "chart_image": f"data:image/png;base64,{chart_image}"
        }
        
    except Exception as e:
        logging.error(f"Error generating OI chart for strike: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

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

@app.get("/api/options/backtest")
async def api_options_backtest(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    strategy_type: str = Query("all", description="Strategy type: 'all', 'single', 'spread', 'multi_leg'"),
    show_only_profitable: bool = Query(False, description="Show only profitable trades"),
    min_profit: Optional[float] = Query(None, description="Minimum profit threshold"),
    timeframe_minutes: int = Query(15, description="Timeframe in minutes (default: 15)"),
    save_results: bool = Query(False, description="Save results to database"),
    force_refresh: bool = Query(False, description="Force refresh - regenerate data even if cached")
):
    """
    API endpoint to run options back-testing for a date range
    """
    try:
        from options.OptionsBacktester import OptionsBacktester
        from datetime import datetime
        
        # Parse dates
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            return {
                "success": False,
                "error": "Invalid date format. Use YYYY-MM-DD"
            }
        
        # Validate date range
        if start_date_obj > end_date_obj:
            return {
                "success": False,
                "error": "Start date must be before end date"
            }
        
        # Initialize back-tester
        backtester = OptionsBacktester()
        
        # Check if results already exist in database (unless force refresh is requested)
        existing_results = None
        if not force_refresh:
            logging.info(f"Checking for cached backtest: start_date={start_date_obj}, end_date={end_date_obj}, "
                        f"strategy_type={strategy_type}, timeframe_minutes={timeframe_minutes}, "
                        f"show_only_profitable={show_only_profitable}, min_profit={min_profit}")
            existing_results = backtester.get_existing_backtest(
                start_date=start_date_obj,
                end_date=end_date_obj,
                strategy_type=strategy_type,
                show_only_profitable=show_only_profitable,
                min_profit=min_profit,
                timeframe_minutes=timeframe_minutes
            )
        
        if existing_results:
            # Return existing results from database
            logging.info(f"Returning cached backtest results (ID: {existing_results.get('backtest_id')}, "
                        f"from_cache={existing_results.get('from_cache', False)})")
            return existing_results
        
        # Run new back-test if no existing results found
        logging.info(f"Running new backtest for {start_date_obj} to {end_date_obj}")
        results = backtester.run_backtest(
            start_date=start_date_obj,
            end_date=end_date_obj,
            strategy_type=strategy_type,
            show_only_profitable=show_only_profitable,
            min_profit=min_profit,
            timeframe_minutes=timeframe_minutes
        )
        
        # Convert date objects to strings in trades and sanitize for JSON
        from datetime import date
        import math
        
        def sanitize_value(val):
            """Sanitize a value for JSON serialization"""
            if val is None:
                return None
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return None
                return val
            if isinstance(val, dict):
                return {k: sanitize_value(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return [sanitize_value(item) for item in val]
            return val
        
        # Sanitize results before returning
        if results.get('success'):
            # Convert dates to strings
            for trade in results.get('trades', []):
                if 'expiry' in trade and isinstance(trade['expiry'], date):
                    trade['expiry'] = trade['expiry'].strftime('%Y-%m-%d')
            
            # Sanitize all float values
            results = sanitize_value(results)
        
        # Always save new results to database to enable caching for future requests
        # This ensures data is stored and can be retrieved without regeneration
        if results.get('success') and not results.get('from_cache'):
            try:
                backtest_id = backtester.save_backtest_results(results=results)
                if backtest_id:
                    results['backtest_id'] = backtest_id
                    results['saved'] = True
                    logging.info(f"Saved backtest results with ID: {backtest_id} for caching")
                else:
                    logging.warning("Failed to save backtest results to database")
            except Exception as e:
                logging.error(f"Error saving backtest results: {e}")
                # Continue even if save fails - results are still returned
        
        return results
        
    except Exception as e:
        logging.error(f"Error running options back-test: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/options/backtest/results")
async def api_options_backtest_results(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    show_only_profitable: bool = Query(False, description="Show only profitable trades"),
    min_profit: Optional[float] = Query(None, description="Minimum profit threshold")
):
    """
    API endpoint to get back-testing results with detailed metrics
    """
    try:
        from options.OptionsBacktester import OptionsBacktester
        from datetime import datetime
        
        # Parse dates
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            return {
                "success": False,
                "error": "Invalid date format. Use YYYY-MM-DD"
            }
        
        # Initialize back-tester
        backtester = OptionsBacktester()
        
        # Run back-test
        results = backtester.run_backtest(
            start_date=start_date_obj,
            end_date=end_date_obj,
            show_only_profitable=show_only_profitable,
            min_profit=min_profit
        )
        
        if not results.get('success'):
            return results
        
        # Convert date objects to strings and sanitize for JSON
        from datetime import date
        import math
        
        def sanitize_value(val):
            """Sanitize a value for JSON serialization"""
            if val is None:
                return None
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return None
                return val
            if isinstance(val, dict):
                return {k: sanitize_value(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return [sanitize_value(item) for item in val]
            return val
        
        # Convert dates and sanitize
        for trade in results.get('trades', []):
            if 'expiry' in trade and isinstance(trade['expiry'], date):
                trade['expiry'] = trade['expiry'].strftime('%Y-%m-%d')
        
        # Sanitize all results
        sanitized_results = sanitize_value({
            "success": True,
            "trades": results.get('trades', []),
            "metrics": results.get('metrics', {}),
            "summary": results.get('summary', {})
        })
        
        return sanitized_results
        
    except Exception as e:
        logging.error(f"Error getting back-testing results: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/options/backtest/payoff_chart")
async def api_options_backtest_payoff_chart(
    entry_date: str = Query(..., description="Entry date in YYYY-MM-DD format"),
    exit_date: str = Query(..., description="Exit date in YYYY-MM-DD format"),
    strike: float = Query(..., description="Strike price"),
    option_type: str = Query(..., description="Option type: 'CE' or 'PE'"),
    entry_price: float = Query(..., description="Entry price"),
    exit_price: float = Query(..., description="Exit price"),
    expiry: str = Query(None, description="Expiry date in YYYY-MM-DD format")
):
    """
    API endpoint to generate payoff chart for a back-tested trade
    """
    try:
        from options.OptionsStrategies import OptionsStrategies
        from datetime import datetime
        
        # Parse dates
        try:
            entry_date_obj = datetime.strptime(entry_date, '%Y-%m-%d').date()
            exit_date_obj = datetime.strptime(exit_date, '%Y-%m-%d').date()
            expiry_date_obj = None
            if expiry:
                expiry_date_obj = datetime.strptime(expiry, '%Y-%m-%d').date()
        except ValueError:
            return {
                "success": False,
                "error": "Invalid date format. Use YYYY-MM-DD"
            }
        
        # Validate option type
        if option_type not in ['CE', 'PE']:
            return {
                "success": False,
                "error": "option_type must be 'CE' or 'PE'"
            }
        
        # Create strategy legs
        strategy_legs = [{
            'action': 'BUY',
            'quantity': 1,
            'option_type': option_type,
            'strike': strike,
            'premium': entry_price,
            'expiry': expiry_date_obj or exit_date_obj
        }]
        
        # Initialize strategies generator
        strategies = OptionsStrategies()
        
        # Get underlying prices
        from options.HistoricalOptionsGenerator import HistoricalOptionsGenerator
        generator = HistoricalOptionsGenerator()
        entry_underlying = generator._get_underlying_price(entry_date_obj)
        exit_underlying = generator._get_underlying_price(exit_date_obj)
        
        if entry_underlying is None or exit_underlying is None:
            return {
                "success": False,
                "error": "Could not fetch underlying prices for the dates"
            }
        
        # Generate chart
        chart_image = strategies.generate_historical_payoff_chart(
            strategy_legs=strategy_legs,
            entry_date=entry_date_obj,
            exit_date=exit_date_obj,
            entry_underlying_price=entry_underlying,
            exit_underlying_price=exit_underlying
        )
        
        if not chart_image:
            return {
                "success": False,
                "error": "Failed to generate chart"
            }
        
        return {
            "success": True,
            "chart_image": chart_image
        }
        
    except Exception as e:
        logging.error(f"Error generating payoff chart: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }
@app.get("/api/options/historical_chain")
async def api_options_historical_chain(
    analysis_date: str = Query(..., description="Analysis date in YYYY-MM-DD format"),
    expiry: str = Query(None, description="Expiry date in YYYY-MM-DD format"),
    strike_min: Optional[float] = Query(None, description="Minimum strike price"),
    strike_max: Optional[float] = Query(None, description="Maximum strike price"),
    option_type: str = Query(None, description="Option type: 'CE' or 'PE'")
):
    """
    API endpoint to get historical options chain for a specific date
    """
    try:
        from options.HistoricalOptionsGenerator import HistoricalOptionsGenerator
        from datetime import datetime
        import pandas as pd
        
        # Parse dates
        try:
            analysis_date_obj = datetime.strptime(analysis_date, '%Y-%m-%d').date()
            expiry_date_obj = None
            if expiry:
                expiry_date_obj = datetime.strptime(expiry, '%Y-%m-%d').date()
        except ValueError:
            return {
                "success": False,
                "error": "Invalid date format. Use YYYY-MM-DD"
            }
        
        # Validate option type
        if option_type and option_type not in ['CE', 'PE']:
            return {
                "success": False,
                "error": "option_type must be 'CE' or 'PE'"
            }
        
        # Initialize generator
        generator = HistoricalOptionsGenerator()
        
        # Determine strike range
        strike_range = None
        if strike_min is not None and strike_max is not None:
            strike_range = (strike_min, strike_max)
        
        # Get historical options chain
        chain_df = generator.get_historical_options_chain(
            analysis_date=analysis_date_obj,
            expiry_date=expiry_date_obj,
            strike_range=strike_range,
            option_type=option_type
        )
        
        if chain_df.empty:
            return {
                "success": True,
                "options_chain": [],
                "total_options": 0,
                "data_source": "No data available"
            }
        
        # Convert to list of dictionaries
        options_list = []
        for _, row in chain_df.iterrows():
            option_dict = {
                'instrument_token': int(row.get('instrument_token', 0)),
                'tradingsymbol': str(row.get('tradingsymbol', '')),
                'strike_price': float(row.get('strike_price', 0)),
                'option_type': str(row.get('option_type', '')),
                'expiry': row.get('expiry').strftime('%Y-%m-%d') if pd.notna(row.get('expiry')) else None,
                'last_price': float(row.get('last_price', 0)),
                'volume': int(row.get('volume', 0)),
                'oi': int(row.get('oi', 0)),
                'average_price': float(row.get('average_price', 0)),
                'is_generated': bool(row.get('is_generated', False)),
                'data_source': str(row.get('data_source', 'unknown'))
            }
            options_list.append(option_dict)
        
        # Count data sources
        generated_count = chain_df['is_generated'].sum() if 'is_generated' in chain_df.columns else 0
        historical_count = len(chain_df) - generated_count
        
        return {
            "success": True,
            "options_chain": options_list,
            "total_options": len(options_list),
            "data_source": {
                "generated_count": int(generated_count),
                "historical_count": int(historical_count),
                "generated_percentage": round((generated_count / len(chain_df) * 100) if len(chain_df) > 0 else 0, 2),
                "historical_percentage": round((historical_count / len(chain_df) * 100) if len(chain_df) > 0 else 0, 2)
            },
            "analysis_date": analysis_date
        }
        
    except Exception as e:
        logging.error(f"Error getting historical options chain: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/options/backtest/saved")
async def api_options_backtest_saved(
    limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    start_date: Optional[str] = Query(None, description="Filter by start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter by end date (YYYY-MM-DD)"),
    strategy_type: Optional[str] = Query(None, description="Filter by strategy type")
):
    """
    API endpoint to retrieve saved back-testing results
    """
    try:
        from common.Boilerplate import get_db_connection
        from datetime import datetime
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build WHERE clause
        where_conditions = []
        params = []
        
        if start_date:
            try:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
                where_conditions.append("start_date >= %s")
                params.append(start_date_obj)
            except ValueError:
                pass
        
        if end_date:
            try:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
                where_conditions.append("end_date <= %s")
                params.append(end_date_obj)
            except ValueError:
                pass
        
        if strategy_type:
            where_conditions.append("strategy_type = %s")
            params.append(strategy_type)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM my_schema.options_backtest_results WHERE {where_clause}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]
        
        # Get results
        query = f"""
            SELECT 
                id, backtest_name, start_date, end_date, strategy_type,
                total_trades, win_rate, total_profit_loss, sharpe_ratio,
                confidence_score, created_at
            FROM my_schema.options_backtest_results
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        
        params.extend([limit, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'backtest_name': row[1],
                'start_date': row[2].strftime('%Y-%m-%d') if row[2] else None,
                'end_date': row[3].strftime('%Y-%m-%d') if row[3] else None,
                'strategy_type': row[4],
                'total_trades': row[5],
                'win_rate': float(row[6]) if row[6] else 0.0,
                'total_profit_loss': float(row[7]) if row[7] else 0.0,
                'sharpe_ratio': float(row[8]) if row[8] else 0.0,
                'confidence_score': float(row[9]) if row[9] else 0.0,
                'created_at': row[10].isoformat() if row[10] else None
            })
        
        return {
            "success": True,
            "results": results,
            "total": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logging.error(f"Error retrieving saved back-test results: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/options/backtest/saved/{backtest_id}")
async def api_options_backtest_saved_detail(backtest_id: int):
    """
    API endpoint to get detailed saved back-testing result by ID
    """
    try:
        from common.Boilerplate import get_db_connection
        import json
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get main result
        query = """
            SELECT 
                id, backtest_name, start_date, end_date, strategy_type, timeframe_minutes,
                show_only_profitable, min_profit, total_trades, win_count, loss_count,
                win_rate, total_profit_loss, avg_profit_loss, gross_profit, gross_loss,
                avg_win, avg_loss, max_profit, max_loss, profit_factor, sharpe_ratio,
                max_drawdown, avg_holding_period, confidence_score, data_quality, metrics, summary,
                created_at
            FROM my_schema.options_backtest_results
            WHERE id = %s
        """
        
        cursor.execute(query, (backtest_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {
                "success": False,
                "error": f"Back-test result with ID {backtest_id} not found"
            }
        
        # Get trades
        trades_query = """
            SELECT 
                entry_date, exit_date, symbol, option_type, strike_price, expiry,
                entry_price, exit_price, profit_loss, exit_reason, holding_period,
                is_generated, data_source, trade_details
            FROM my_schema.options_backtest_trades
            WHERE backtest_result_id = %s
            ORDER BY entry_date, exit_date
        """
        
        cursor.execute(trades_query, (backtest_id,))
        trade_rows = cursor.fetchall()
        conn.close()
        
        trades = []
        for trade_row in trade_rows:
            trade_details = json.loads(trade_row[13]) if trade_row[13] else {}
            trades.append({
                'entry_date': trade_row[0].strftime('%Y-%m-%d') if trade_row[0] else None,
                'exit_date': trade_row[1].strftime('%Y-%m-%d') if trade_row[1] else None,
                'symbol': trade_row[2],
                'option_type': trade_row[3],
                'strike_price': float(trade_row[4]) if trade_row[4] else None,
                'expiry': trade_row[5].strftime('%Y-%m-%d') if trade_row[5] else None,
                'entry_price': float(trade_row[6]) if trade_row[6] else None,
                'exit_price': float(trade_row[7]) if trade_row[7] else None,
                'profit_loss': float(trade_row[8]) if trade_row[8] else None,
                'exit_reason': trade_row[9],
                'holding_period': trade_row[10],
                'is_generated': trade_row[11],
                'data_source': trade_row[12],
                **trade_details
            })
        
        # Parse JSON fields
        data_quality = json.loads(row[25]) if row[25] else {}
        metrics = json.loads(row[26]) if row[26] else {}
        summary = json.loads(row[27]) if row[27] else {}
        
        return {
            "success": True,
            "id": row[0],
            "backtest_name": row[1],
            "start_date": row[2].strftime('%Y-%m-%d') if row[2] else None,
            "end_date": row[3].strftime('%Y-%m-%d') if row[3] else None,
            "strategy_type": row[4],
            "timeframe_minutes": row[5],
            "filters": {
                "show_only_profitable": row[6],
                "min_profit": float(row[7]) if row[7] else None
            },
            "metrics": {
                "total_trades": row[8],
                "win_count": row[9],
                "loss_count": row[10],
                "win_rate": float(row[11]) if row[11] else 0.0,
                "total_profit_loss": float(row[12]) if row[12] else 0.0,
                "avg_profit_loss": float(row[13]) if row[13] else 0.0,
                "gross_profit": float(row[14]) if row[14] else 0.0,
                "gross_loss": float(row[15]) if row[15] else 0.0,
                "avg_win": float(row[16]) if row[16] else 0.0,
                "avg_loss": float(row[17]) if row[17] else 0.0,
                "max_profit": float(row[18]) if row[18] else 0.0,
                "max_loss": float(row[19]) if row[19] else 0.0,
                "profit_factor": float(row[20]) if row[20] else None,
                "sharpe_ratio": float(row[21]) if row[21] else 0.0,
                "max_drawdown": float(row[22]) if row[22] else 0.0,
                "avg_holding_period": float(row[23]) if row[23] else 0.0,
                "confidence_score": float(row[24]) if row[24] else 0.0,
                "data_quality": data_quality
            },
            "summary": summary,
            "trades": trades,
            "created_at": row[28].isoformat() if row[28] else None
        }
        
    except Exception as e:
        logging.error(f"Error retrieving back-test result detail: {e}")
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

    