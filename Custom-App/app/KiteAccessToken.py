from Boilerplate import *
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import os
from datetime import datetime, timedelta
import json


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

# Global ANALYSIS_DATE configuration - can be set via API or environment
ANALYSIS_DATE = None  # Set to None for current date, or specify date like '2025-10-20'

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
        
        conn.close()
        
        return {
            'active': recent_ticks > 0,
            'recent_ticks': recent_ticks,
            'total_ticks': total_ticks
        }
    except Exception as e:
        logging.error(f"Error checking tick data status: {e}")
        return {
            'active': False,
            'recent_ticks': 0,
            'total_ticks': 0
        }


# Function to get holdings data
def get_holdings_data(page: int = 1, per_page: int = 10, sort_by: str = None, sort_dir: str = "asc"):
    """Fetch current holdings from the database with pagination and sorting."""
    try:
        conn = get_db_connection()
        # Use RealDictCursor to get results as dictionaries
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Fetch holdings for the most recent run_date
        # Get total count for pagination
        cursor.execute("""
            SELECT COUNT(*) as total_count
            FROM my_schema.holdings
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
        """)
        total_count = cursor.fetchone()['total_count']

        # Validate and set sort column
        valid_sort_columns = ['trading_symbol', 'invested_amount', 'current_amount', 'pnl']
        if sort_by not in valid_sort_columns:
            sort_by = 'trading_symbol'
        
        # Validate sort direction
        if sort_dir.lower() not in ['asc', 'desc']:
            sort_dir = 'asc'
        
        # Calculate offset for pagination
        offset = (page - 1) * per_page

        # Fetch paginated holdings for the most recent run_date
        # Calculate invested amount and current value
        cursor.execute("""
            SELECT 
                trading_symbol,
                instrument_token,
                quantity,
                average_price,
                last_price,
                pnl,
                (quantity * average_price) as invested_amount,
                (quantity * last_price) as current_amount,
                (SELECT MAX(run_date) FROM my_schema.holdings) as run_date
            FROM my_schema.holdings
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ORDER BY {sort_by} {sort_dir}
            LIMIT %s OFFSET %s
        """.format(sort_by=sort_by, sort_dir=sort_dir.upper()), (per_page, offset))
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
    
    return {
        'token_fetched_today': is_token_fetched_today(),
        'token_valid': is_token_currently_valid(),
        'tick_data': get_tick_data_status(),
        'last_update': ist_now.strftime('%H:%M:%S IST')
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
        
        # Get today's P&L for all holdings
        today_pnl_map = {}
        if prev_date:
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT DISTINCT trading_symbol, quantity
                    FROM my_schema.holdings
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                )
                SELECT 
                    h.trading_symbol,
                    h.quantity,
                    COALESCE(today_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    h.quantity * (COALESCE(today_p.price_close, 0) - COALESCE(prev_p.price_close, 0)) as today_pnl
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date))
            
            for row in cursor.fetchall():
                today_price = float(row['today_price']) if row['today_price'] else 0.0
                prev_price = float(row['prev_price']) if row['prev_price'] else 0.0
                today_pnl = float(row['today_pnl']) if row['today_pnl'] else 0.0
                
                # Calculate percentage change
                pct_change = 0.0
                if prev_price > 0:
                    pct_change = ((today_price - prev_price) / prev_price) * 100
                
                today_pnl_map[row['trading_symbol']] = {
                    'today_pnl': today_pnl,
                    'today_price': today_price,
                    'prev_price': prev_price,
                    'pct_change': pct_change
                }
        
        conn.close()
        
        # Convert RealDictRow objects to regular dictionaries and enrich with today's P&L
        enriched_holdings = []
        for holding in holdings_data.get('holdings', []):
            # Convert RealDictRow to dict
            holding_dict = dict(holding)
            symbol = holding_dict.get('trading_symbol', '')
            today_pnl_info = today_pnl_map.get(symbol, {'today_pnl': 0.0, 'today_price': 0.0, 'prev_price': 0.0, 'pct_change': 0.0})
            
            # Add today's P&L fields
            holding_dict['today_pnl'] = today_pnl_info['today_pnl']
            holding_dict['today_price'] = today_pnl_info['today_price']
            holding_dict['prev_price'] = today_pnl_info['prev_price']
            holding_dict['pct_change'] = today_pnl_info.get('pct_change', 0.0)
            
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
            holdings_info = get_holdings_data(page=page, per_page=10)
            
            # Enrich holdings with today's P&L
            holdings_info = enrich_holdings_with_today_pnl(holdings_info)
            
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "token_status": "Valid" if system_status['token_valid'] else "Invalid",
                "tick_status": "Active" if tick_data['active'] else "Inactive",
                "last_update": system_status['last_update'],
                "total_ticks": tick_data['total_ticks'],
                "holdings_info": holdings_info
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
    
    if request_token and status == "success" and action == "login" and type == "login":
        # Handle redirect from Zerodha with request_token
        redis_client.set("kite_request_token", request_token)
        logging.info(f"Received request_token: {request_token}")
        token_access = request_token

        # Trigger get_access_token to generate and save access token
        access_token = get_access_token()  # Call your function here

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
            "holdings_info": holdings_info
        })
    else:
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

@app.get("/refresh_data")
async def refresh_data():
    """Endpoint to refresh data"""
    return {"message": "Data refresh initiated", "status": "success"}

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
    ACCESS_TOKEN = ''
    
    # Check if existing access token is provided and valid

    #ACCESS_TOKEN = redis_client.get("kite_access_token")
    #if is_access_token_valid(ACCESS_TOKEN):
    #    return ACCESS_TOKEN
    #else:
    # Clear any existing request_token in Redis
    #redis_client.delete("kite_request_token")
    redis_client.delete("kite_access_token")
    redis_client.delete("kite_access_token_timestamp")
    print('2')
    
    # Wait for request_token from Redis
    timeout = 600  # 10 minutes
    start_time = time.time()
    while time.time() - start_time < timeout:
        request_token = redis_client.get("kite_request_token")
        if request_token:
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

@app.get("/api/holdings")
async def api_holdings(page: int = Query(1, ge=1), per_page: int = Query(10, ge=1), sort_by: str = Query(None), sort_dir: str = Query("asc")):
    """API endpoint to get paginated holdings data with sorting and GTT info"""
    try:
        holdings_info = get_holdings_data(page=page, per_page=per_page, sort_by=sort_by, sort_dir=sort_dir)
        
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
        
        # Create a map of today's P&L per stock
        today_pnl_map = {}
        if prev_date:
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT DISTINCT trading_symbol, quantity
                    FROM my_schema.holdings
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                )
                SELECT 
                    h.trading_symbol,
                    h.quantity,
                    COALESCE(today_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    h.quantity * (COALESCE(today_p.price_close, 0) - COALESCE(prev_p.price_close, 0)) as today_pnl
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date))
            
            for row in cursor.fetchall():
                today_price = float(row['today_price']) if row['today_price'] else 0.0
                prev_price = float(row['prev_price']) if row['prev_price'] else 0.0
                today_pnl = float(row['today_pnl']) if row['today_pnl'] else 0.0
                
                # Calculate percentage change
                pct_change = 0.0
                if prev_price > 0:
                    pct_change = ((today_price - prev_price) / prev_price) * 100
                
                today_pnl_map[row['trading_symbol']] = {
                    'today_pnl': today_pnl,
                    'today_price': today_price,
                    'prev_price': prev_price,
                    'pct_change': pct_change
                }
        
        conn.close()
        
        # Convert holdings to serializable format
        holdings_list = []
        for holding in holdings_info["holdings"]:
            symbol = holding["trading_symbol"]
            today_pnl_info = today_pnl_map.get(symbol, {'today_pnl': 0.0, 'today_price': 0.0, 'prev_price': 0.0, 'pct_change': 0.0})
            
            holdings_list.append({
                "trading_symbol": symbol,
                "instrument_token": holding["instrument_token"],
                "quantity": holding["quantity"],
                "average_price": float(holding["average_price"]) if holding["average_price"] else 0.0,
                "last_price": float(holding["last_price"]) if holding["last_price"] else 0.0,
                "pnl": float(holding["pnl"]) if holding["pnl"] else 0.0,
                "invested_amount": float(holding.get("invested_amount", 0)),
                "current_amount": float(holding.get("current_amount", 0)),
                "today_pnl": today_pnl_info['today_pnl'],
                "today_price": today_pnl_info['today_price'],
                "prev_price": today_pnl_info['prev_price'],
                "pct_change": today_pnl_info.get('pct_change', 0.0),
                "existing_gtt": gtt_map.get(symbol)
            })
        
        return {
            "holdings": holdings_list,
            "total_count": holdings_info["total_count"],
            "page": holdings_info["page"],
            "per_page": holdings_info["per_page"]
        }
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
            positions_list.append({
                "fetch_timestamp": position["fetch_timestamp"].strftime("%Y-%m-%d %H:%M:%S") if position["fetch_timestamp"] else "",
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

@app.get("/api/futures_order_flow")
async def api_futures_order_flow():
    """API endpoint to get Nifty 50 futures order flow data - always shows most recent trading day"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the last 5 orders from futures_tick_depth for the most recent trading date (weekday)
        # Always shows the most recent weekday with data, regardless of analysis date
        cursor.execute("""
            WITH last_trading_date AS (
                SELECT MAX(run_date) as last_date
                FROM my_schema.futures_tick_depth 
                WHERE EXTRACT(DOW FROM run_date) BETWEEN 1 AND 5  -- Monday(1) to Friday(5)
            )
            SELECT 
                ftd.timestamp,
                ftd.side,
                ftd.price,
                ftd.quantity,
                ftd.orders,
                ftd.run_date
            FROM my_schema.futures_tick_depth ftd
            CROSS JOIN last_trading_date ltd
            WHERE ftd.run_date = ltd.last_date
            ORDER BY ftd.timestamp DESC
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        # Format the results
        order_flow_data = []
        for row in results:
            order_flow_data.append({
                "timestamp": row[0].strftime("%H:%M:%S") if row[0] else "",
                "side": row[1],
                "price": float(row[2]) if row[2] else 0.0,
                "quantity": int(row[3]) if row[3] else 0,
                "orders": int(row[4]) if row[4] else 0,
                "run_date": row[5].strftime("%Y-%m-%d") if row[5] else ""
            })
        
        return {
            "order_flow": order_flow_data,
            "total_orders": len(order_flow_data),
            "trading_date": order_flow_data[0]['run_date'] if order_flow_data else None
        }
    except Exception as e:
        logging.error(f"Error fetching futures order flow: {e}")
        return {"error": str(e)}
