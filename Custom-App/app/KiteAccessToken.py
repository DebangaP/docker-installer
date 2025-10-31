from Boilerplate import *
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Request, Query
from fastapi.responses import StreamingResponse, FileResponse
import os
from datetime import datetime, timedelta
import json
import io
import psycopg2
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
        valid_sort_columns = ['trading_symbol', 'invested_amount', 'current_amount', 'pnl', 'today_pnl']
        if sort_by not in valid_sort_columns:
            sort_by = 'trading_symbol'
        
        # Validate sort direction
        if sort_dir.lower() not in ['asc', 'desc']:
            sort_dir = 'asc'

        # Calculate offset for pagination
        offset = (page - 1) * per_page

        # Fetch paginated holdings for the most recent run_date
        # Calculate invested amount and current value with coalesced prices
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
                (SELECT MAX(run_date) FROM my_schema.holdings) as run_date
            FROM my_schema.holdings h
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE price_date::date <= CURRENT_DATE
                ORDER BY scrip_id, price_date DESC
            ) rt ON h.trading_symbol = rt.scrip_id
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
        
        # Get today's date
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        # Get previous trading day
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
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
                
            """, (today_str, prev_date_str))
            
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
        
        conn.close()
        
        # Convert RealDictRow objects to regular dictionaries and enrich with today's P&L
        enriched_holdings = []
        for holding in holdings_data.get('holdings', []):
            # Convert RealDictRow to dict
            holding_dict = dict(holding)
            instrument_token = holding_dict.get('instrument_token')
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'today_price': 0.0, 'prev_price': 0.0, 'pct_change': 0.0, 'today_change': '0 (0%)'})
            
            # Add today's P&L fields
            holding_dict['today_pnl'] = today_pnl_info['today_pnl']
            holding_dict['today_price'] = today_pnl_info['today_price']
            holding_dict['prev_price'] = today_pnl_info['prev_price']
            holding_dict['pct_change'] = today_pnl_info.get('pct_change', 0.0)
            holding_dict['today_change'] = today_pnl_info.get('today_change', '0 (0%)')
            
            # Ensure pnl_pct_change field exists (from original query)
            if 'pnl_pct_change' not in holding_dict:
                # Calculate it if missing
                invested_amount = holding_dict.get('invested_amount', 0)
                pnl = holding_dict.get('pnl', 0)
                holding_dict['pnl_pct_change'] = (pnl / invested_amount * 100) if invested_amount != 0 else 0.0
            
            enriched_holdings.append(holding_dict)
        
        # Update holdings_data with enriched holdings
        holdings_data['holdings'] = enriched_holdings
        return holdings_data
        
    except Exception as e:
        logging.error(f"Error enriching holdings with today's P&L: {e}")
        # Return holdings with default today_pnl values
        enriched_holdings = []
        for holding in holdings_data.get('holdings', []):
            holding_dict = dict(holding)
            holding_dict['today_pnl'] = 0.0
            holding_dict['today_price'] = 0.0
            holding_dict['prev_price'] = 0.0
            holding_dict['pct_change'] = 0.0
            holding_dict['today_change'] = '0 (0%)'
            
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
    return {
        "token_valid": status['token_valid'],
        "token_fetched_today": status['token_fetched_today'],
        "tick_active": status['tick_data']['active'],
        "recent_ticks": status['tick_data']['recent_ticks'],
        "total_ticks": status['tick_data']['total_ticks'],
        "last_update": status['last_update']
    }

@app.get("/api/tpo_charts")
async def api_tpo_charts(analysis_date: str = Query(None)):
    """API endpoint to generate TPO chart images"""
    try:
        from CalculateTPO import PostgresDataFetcher, TPOProfile
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
        
        # Generate charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Pre-market chart
        if not pre_market_df.empty:
            pre_market_tpo = TPOProfile(tick_size=5)
            pre_market_tpo.calculate_tpo(pre_market_df)
            pre_market_tpo.plot_profile(ax=ax1, show_metrics=True, show_letters=True)
            ax1.set_title(f"Pre-market TPO Profile ({target_date})", fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No pre-market data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f"Pre-market TPO Profile ({target_date})", fontsize=14, fontweight='bold')
        
        # Real-time chart
        if not real_time_df.empty:
            real_time_tpo = TPOProfile(tick_size=5)
            real_time_tpo.calculate_tpo(real_time_df)
            real_time_tpo.plot_profile(ax=ax2, show_metrics=True, show_letters=True)
            ax2.set_title(f"Real-time TPO Profile ({target_date})", fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No real-time data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f"Real-time TPO Profile ({target_date})", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
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
        from KiteFetchFuture import fetch_and_save_futures
        result = fetch_and_save_futures()
        return result
    except Exception as e:
        logging.error(f"Error refreshing futures data: {e}")
        return {"success": False, "error": str(e)}

@app.get("/refresh_data")
async def refresh_data():
    """Endpoint to refresh all data including futures"""
    try:
        from KiteFetchFuture import fetch_and_save_futures
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
        from MarketBiasAnalyzer import MarketBiasAnalyzer, PostgresDataFetcher
        
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
        from MarketBiasAnalyzer import MarketBiasAnalyzer, PostgresDataFetcher
        
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
        from PremarketAnalyzer import PremarketAnalyzer
        from CalculateTPO import PostgresDataFetcher
        
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
        
        return analysis
    except Exception as e:
        logging.error(f"Error generating pre-market analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.get("/api/gainers")
async def api_gainers():
    """API endpoint to get top 15 gainers from last trading day"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            select "Curr".scrip_id, 
                   100*("Curr".price_close - "Prev".price_close)/"Prev".price_close "Gain",
                   "Curr".price_close as current_price,
                   "Prev".price_close as previous_price
            from my_schema.master_scrips ms,
            (
                select SCRIP_ID, PRICE_CLOSE from my_schema.rt_intraday_price rip 
                where PRICE_DATE = (select mAX(PRICE_DATE) from my_schema.rt_intraday_price rip2 where country = 'IN' 
                                    and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH'))
                and country = 'IN'
                and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
            ) "Curr",
            (
                select SCRIP_ID, PRICE_CLOSE from my_schema.rt_intraday_price rip 
                where PRICE_DATE = (select DATE(mAX(PRICE_DATE))::text from my_schema.rt_intraday_price rip2 
                                    where price_date < (select max(price_date) from my_schema.rt_intraday_price rip5 
                                                        where country = 'IN' 
                                                        and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')) )
                and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                and country = 'IN'
            ) "Prev"
            where "Curr".scrip_id = "Prev".scrip_id
            and "Curr".scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH') 
            and "Curr".scrip_id = ms.scrip_id
            and ms.scrip_country = 'IN'
            order by 100*("Curr".price_close - "Prev".price_close)/"Prev".price_close desc
            limit 15
        """)
        
        gainers = cursor.fetchall()
        gainers_list = []
        for row in gainers:
            gainers_list.append({
                "scrip_id": row['scrip_id'],
                "gain": float(row['Gain']) if row['Gain'] else 0.0,
                "current_price": float(row['current_price']) if row['current_price'] else 0.0,
                "previous_price": float(row['previous_price']) if row['previous_price'] else 0.0
            })
        
        conn.close()
        return {"gainers": gainers_list}
    except Exception as e:
        logging.error(f"Error fetching gainers: {e}")
        return {"error": str(e), "gainers": []}

@app.get("/api/losers")
async def api_losers():
    """API endpoint to get top 15 losers from last trading day"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            select "Curr".scrip_id, 
                   100*("Curr".price_close - "Prev".price_close)/"Prev".price_close "Gain",
                   "Curr".price_close as current_price,
                   "Prev".price_close as previous_price
            from my_schema.master_scrips ms,
            (
                select SCRIP_ID, PRICE_CLOSE from my_schema.rt_intraday_price rip 
                where PRICE_DATE = (select mAX(PRICE_DATE) from my_schema.rt_intraday_price rip2 where country = 'IN' 
                                    and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH'))
                and country = 'IN'
                and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
            ) "Curr",
            (
                select SCRIP_ID, PRICE_CLOSE from my_schema.rt_intraday_price rip 
                where PRICE_DATE = (select DATE(mAX(PRICE_DATE))::text from my_schema.rt_intraday_price rip2 
                                    where price_date < (select max(price_date) from my_schema.rt_intraday_price rip5 
                                                        where country = 'IN' 
                                                        and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')) )
                and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                and country = 'IN'
            ) "Prev"
            where "Curr".scrip_id = "Prev".scrip_id
            and "Curr".scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH') 
            and "Curr".scrip_id = ms.scrip_id
            and ms.scrip_country = 'IN'
            order by 100*("Prev".price_close - "Curr".price_close)/"Prev".price_close desc
            limit 15
        """)
        
        losers = cursor.fetchall()
        losers_list = []
        for row in losers:
            losers_list.append({
                "scrip_id": row['scrip_id'],
                "gain": float(row['Gain']) if row['Gain'] else 0.0,
                "current_price": float(row['current_price']) if row['current_price'] else 0.0,
                "previous_price": float(row['previous_price']) if row['previous_price'] else 0.0
            })
        
        conn.close()
        return {"losers": losers_list}
    except Exception as e:
        logging.error(f"Error fetching losers: {e}")
        return {"error": str(e), "losers": []}

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
        # If sorting by today_pnl, we need to get all holdings, enrich them, sort, then paginate
        if sort_by == 'today_pnl':
            # Get all holdings without pagination
            holdings_info = get_holdings_data(page=1, per_page=10000, sort_by='trading_symbol', sort_dir='asc', search=search)
        else:
            holdings_info = get_holdings_data(page=page, per_page=per_page, sort_by=sort_by, sort_dir=sort_dir, search=search)
        
        # Get all active GTTs for holdings lookup
        from KiteGTT import KiteGTTManager
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
        
        conn.close()
        
        # Convert holdings to serializable format
        holdings_list = []
        for holding in holdings_info["holdings"]:
            symbol = holding["trading_symbol"]
            instrument_token = holding["instrument_token"]
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'today_price': 0.0, 'prev_price': 0.0, 'pct_change': 0.0, 'today_change': '0 (0%)'})
            
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
                "existing_gtt": gtt_map.get(symbol)
            })
        
        # If sorting by today_pnl, sort the enriched list and then paginate
        if sort_by == 'today_pnl':
            sort_reverse = sort_dir.lower() == 'desc'
            holdings_list.sort(key=lambda x: x['today_pnl'], reverse=sort_reverse)
            
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
        return result
    except Exception as e:
        logging.error(f"Error fetching holdings: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}

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
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.positions)
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
        
        return {"positions": positions_list}
    except Exception as e:
        logging.error(f"Error fetching positions: {e}")
        return {"error": str(e), "positions": []}

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
        
        return {"mf_holdings": mf_holdings_list}
    except Exception as e:
        logging.error(f"Error fetching MF holdings: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e), "mf_holdings": []}

@app.get("/api/today_pnl_summary")
async def api_today_pnl_summary():
    """API endpoint to get today's P&L summary for Equity, MF, and Intraday trades"""
    try:
        cached = cache_get_json("today_pnl_summary")
        if cached:
            return cached
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get today's date
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        # Get previous trading day
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
        """, (today_str,))
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        # Calculate Equity P&L - sum up individual holdings P&L
        equity_pnl = 0.0
        if prev_date:
            # Convert prev_date to string format  #
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
            """, (today_str, prev_date_str))
            
            # Sum up all individual holdings P&L
            for row in cursor.fetchall():
                holding_pnl = float(row['today_pnl']) if row['today_pnl'] else 0.0
                equity_pnl += holding_pnl
        
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
        return result
    except Exception as e:
        logging.error(f"Error fetching today's P&L summary: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "equity_pnl": 0.0,
            "equity_overall_pnl": 0.0,
            "mf_pnl": 0.0,
            "mf_overall_pnl": 0.0,
            "intraday_pnl": 0.0,
            "total_today_pnl": 0.0,
            "total_overall_pnl": 0.0,
            "error": str(e)
        }

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
        
        return {"portfolio_history": chart_data}
    except Exception as e:
        logging.error(f"Error fetching portfolio history: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e), "portfolio_history": []}

@app.get("/api/sparkline/{trading_symbol}")
async def api_sparkline(trading_symbol: str, days: int = Query(30, ge=7, le=90)):
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
        
        return {
            "trading_symbol": trading_symbol,
            "prices": prices,
            "labels": labels,
            "min_price": min(prices) if prices else 0,
            "max_price": max(prices) if prices else 0
        }
    except Exception as e:
        logging.error(f"Error fetching sparkline data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e), "prices": [], "labels": []}

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
        
        # Build data array with indicators
        for i in range(len(dates)):
            data.append({
                "date": dates[i],
                "open": opens[i],
                "high": highs[i],
                "low": lows[i],
                "close": closes[i],
                "volume": volumes[i],
                "sma_20": float(sma_20[i]) if not np.isnan(sma_20[i]) else None,
                "sma_50": float(sma_50[i]) if not np.isnan(sma_50[i]) else None,
                "sma_200": float(sma_200[i]) if not np.isnan(sma_200[i]) else None,
                "supertrend": float(supertrend[i]) if not np.isnan(supertrend[i]) else None,
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
        from KiteGTT import KiteGTTManager
        
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
        from KiteGTT import KiteGTTManager
        
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
        from KiteGTT import KiteGTTManager
        
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
        from KiteGTT import KiteGTTManager
        
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
        from KiteGTT import KiteGTTManager
        
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
        from CalculateTPO import PostgresDataFetcher
        from DerivativesTPOAnalyzer import DerivativesTPOAnalyzer
        from DerivativesSuggestionEngine import DerivativesSuggestionEngine
        
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
        
        cursor.execute("""
            SELECT 
                trading_symbol,
                quantity,
                average_price,
                last_price,
                (quantity * average_price) as invested_amount,
                (quantity * last_price) as current_amount,
                pnl
            FROM my_schema.holdings
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ORDER BY trading_symbol
        """)
        
        holdings = cursor.fetchall()
        conn.close()
        
        df = pd.DataFrame([dict(row) for row in holdings])
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
        return {"error": str(e)}

@app.get("/api/download/holdings/csv")
async def download_holdings_csv():
    """Download holdings data as CSV file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                trading_symbol,
                quantity,
                average_price,
                last_price,
                (quantity * average_price) as invested_amount,
                (quantity * last_price) as current_amount,
                pnl
            FROM my_schema.holdings
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ORDER BY trading_symbol
        """)
        
        holdings = cursor.fetchall()
        conn.close()
        
        df = pd.DataFrame([dict(row) for row in holdings])
        output = io.StringIO()
        df.to_csv(output, index=False)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=holdings_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    except Exception as e:
        logging.error(f"Error generating CSV file: {e}")
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
        
        cursor.execute("""
            SELECT 
                trading_symbol, quantity, average_price, last_price,
                (quantity * average_price) as invested_amount,
                (quantity * last_price) as current_amount, pnl
            FROM my_schema.holdings
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ORDER BY trading_symbol
        """)
        
        holdings = cursor.fetchall()
        conn.close()
        
        # Create PDF in memory
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter, topMargin=0.5*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1a1a1a'), spaceAfter=12)
        
        # Calculate totals
        total_invested = sum(row['invested_amount'] for row in holdings)
        total_current = sum(row['current_amount'] for row in holdings)
        total_pnl = sum(row['pnl'] for row in holdings)
        
        elements = []
        elements.append(Paragraph("Equity Holdings Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add summary box
        summary_data = [
            ['Total Invested:', f'Rs {total_invested:,.2f}'],
            ['Current Value:', f'Rs {total_current:,.2f}'],
            ['Total P&L:', f'Rs {total_pnl:,.2f}']
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
        data = [['Symbol', 'Qty', 'Avg Price', 'LTP', 'Invested', 'Current', 'P&L']]
        for row in holdings:
            row_dict = dict(row)
            data.append([
                row_dict['trading_symbol'],
                str(row_dict['quantity']),
                f"Rs {row_dict['average_price']:.2f}",
                f"Rs {row_dict['last_price']:.2f}",
                f"Rs {row_dict['invested_amount']:.2f}",
                f"Rs {row_dict['current_amount']:.2f}",
                f"Rs {row_dict['pnl']:.2f}"
            ])
        
        # Add totals row
        data.append([
            'TOTAL',
            '',
            '',
            '',
            f'Rs {total_invested:,.2f}',
            f'Rs {total_current:,.2f}',
            f'Rs {total_pnl:,.2f}'
        ])
        
        table = Table(data, repeatRows=1, colWidths=[1.5*inch, 0.6*inch, 0.9*inch, 0.9*inch, 1.1*inch, 1.1*inch, 1*inch])
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

@app.get("/api/portfolio_hedge_analysis")
async def api_portfolio_hedge_analysis(
    target_hedge_ratio: float = Query(0.5, ge=0.0, le=1.0, description="Target hedge ratio (0.0 to 1.0)"),
    strategy_type: str = Query("all", description="Strategy type filter: all, puts, calls, collars, futures, or comma-separated")
):
    """API endpoint to get portfolio hedge analysis with multiple strategies"""
    try:
        from PortfolioHedgeAnalyzer import PortfolioHedgeAnalyzer
        
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
        from OptionsScanner import OptionsScanner
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
        
        # Limit results to top 5 for display
        candidates = candidates[:5] if candidates else []
        
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
    except Exception as e:
        logging.error(f"Error scanning options chain: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
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
        from OptionsDataFetcher import OptionsDataFetcher
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
        from OptionsDataFetcher import OptionsDataFetcher
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
        from MarginCalculator import MarginCalculator
        
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
        from MarginCalculator import MarginCalculator
        
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