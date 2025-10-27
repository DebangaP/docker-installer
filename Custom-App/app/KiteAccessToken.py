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

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
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
            
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "token_status": "Valid" if system_status['token_valid'] else "Invalid",
                "tick_status": "Active" if tick_data['active'] else "Inactive",
                "last_update": system_status['last_update'],
                "total_ticks": tick_data['total_ticks']
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
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "token_status": "Valid" if system_status['token_valid'] else "Invalid",
            "tick_status": "Active" if tick_data['active'] else "Inactive",
            "last_update": system_status['last_update'],
            "total_ticks": tick_data['total_ticks']
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

