import os
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException   
import redis
import time
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd

#import InsertOHLC   # run this code to insert latest data

# Configure logging
logging.basicConfig(level=logging.INFO)

token_access = ''

def get_db_connection():
    return psycopg2.connect(
        host="postgres",
        database="mydb",
        user="postgres",
        password="postgres"
    )

conn = get_db_connection()
cursor = conn.cursor()

def init_postgres_conn():
    try:
        cursor = conn.cursor()
        # Create tables not created already, and not part of Schema.sql so that the Postgres container data does not need to be deleted using "docker rm -f $(docker ps -a -q)"
        cursor.execute("""
                CREATE TABLE IF NOT EXISTS my_schema.margins (
                    fetch_timestamp TIMESTAMP default current_timestamp,
                    run_date date default current_date,
                    margin_type VARCHAR(20),
                    enabled BOOLEAN,
                    net FLOAT,
                    available_adhoc_margin FLOAT,
                    available_cash FLOAT,
                    available_opening_balance FLOAT,
                    available_live_balance FLOAT,
                    available_collateral FLOAT,
                    available_intraday_payin FLOAT,
                    utilised_debits FLOAT,
                    utilised_exposure FLOAT,
                    utilised_m2m_realised FLOAT,
                    utilised_m2m_unrealised FLOAT,
                    utilised_option_premium FLOAT,
                    utilised_payout FLOAT,
                    utilised_span FLOAT,
                    utilised_holding_sales FLOAT,
                    utilised_turnover FLOAT,
                    utilised_liquid_collateral FLOAT,
                    utilised_stock_collateral FLOAT,
                    utilised_equity FLOAT,
                    utilised_delivery FLOAT,
                    PRIMARY KEY (fetch_timestamp, margin_type)
                );
            """)
        conn.commit()
        logging.info("Connected to PostgreSQL and initialized tables")
    except Exception as e:
        logging.error(f"Postgres connection failed: {e}")
    

# Initialize Redis client
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

load_dotenv()

try:
    API_KEY = os.getenv("KITE_API_KEY")
    API_SECRET = os.getenv("KITE_API_SECRET")
    #ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

    print(f"API_KEY: {API_KEY}")
    print(f"API_SECRET: {API_SECRET}")
except KeyError as e:
    logging.error(f"Environment variable {e} not set")
    raise

# Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)
#print(kite)
print('1')

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


def fetch_and_save_profile():
    print('fetching profile')
    profile = kite.profile()  # attempt to fetch user profile
    logging.info(f"User profile: {profile}")
    print(type(profile))

    #print(profile)
    cursor.execute("""
                INSERT INTO my_schema.profile (
                    user_id, user_name, email, user_type, broker,
                    products, order_types, exchanges
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fetch_timestamp, user_id) DO UPDATE
                SET user_name = EXCLUDED.user_name,
                    email = EXCLUDED.email,
                    user_type = EXCLUDED.user_type,
                    broker = EXCLUDED.broker,
                    products = EXCLUDED.products,
                    order_types = EXCLUDED.order_types,
                    exchanges = EXCLUDED.exchanges
            """, (
                profile.get('user_id'), profile.get('user_name'),
                profile.get('email'), profile.get('user_type'), profile.get('broker'),
                profile.get('products', []), profile.get('order_types', []),
                profile.get('exchanges', [])
            ))
    conn.commit()


def fetch_and_save_orders():
    print('fetching orders')
    orders = kite.orders()  # save into postgres table with current date
    logging.info(f"User orders: {orders}")
    #print(orders)
    if not orders:
        logging.warning("No orders found")
        return []
    cursor.executemany("""
                INSERT INTO my_schema.orders (
                    order_id, parent_order_id, exchange_order_id, status,
                    status_message, order_type, transaction_type, exchange, trading_symbol,
                    instrument_token, quantity, price, trigger_price, average_price,
                    order_timestamp, exchange_timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (order_id) DO UPDATE
                SET parent_order_id = EXCLUDED.parent_order_id,
                    exchange_order_id = EXCLUDED.exchange_order_id,
                    status = EXCLUDED.status,
                    status_message = EXCLUDED.status_message,
                    order_type = EXCLUDED.order_type,
                    transaction_type = EXCLUDED.transaction_type,
                    exchange = EXCLUDED.exchange,
                    trading_symbol = EXCLUDED.trading_symbol,
                    instrument_token = EXCLUDED.instrument_token,
                    quantity = EXCLUDED.quantity,
                    price = EXCLUDED.price,
                    trigger_price = EXCLUDED.trigger_price,
                    average_price = EXCLUDED.average_price,
                    order_timestamp = EXCLUDED.order_timestamp,
                    exchange_timestamp = EXCLUDED.exchange_timestamp
            """, [
                (
                    order.get('order_id'), order.get('parent_order_id'),
                    order.get('exchange_order_id'), order.get('status'),
                    order.get('status_message'), order.get('order_type'),
                    order.get('transaction_type'), order.get('exchange'),
                    order.get('trading_symbol'), order.get('instrument_token'),
                    order.get('quantity'), order.get('price'), order.get('trigger_price'),
                    order.get('average_price'),
                    pd.to_datetime(order.get('order_timestamp')) if order.get('order_timestamp') else None,
                    pd.to_datetime(order.get('exchange_timestamp')) if order.get('exchange_timestamp') else None
                ) for order in orders
            ])
    conn.commit()


def fetch_and_save_positions():
    print('fetching positions')
    positions = kite.positions()    # save into postgres table with current date
    logging.info(f"User positions: {positions}")
    #print(positions)
    if not positions:
        logging.warning("No positions found")
        return []
            
    # Process both 'net' and 'day' positions
    all_positions = []
    for position_type in ['net', 'day']:
        for pos in positions.get(position_type, []):
            all_positions.append({
                        'position_type': position_type,
                        'trading_symbol': pos.get('trading_symbol'),
                        'instrument_token': pos.get('instrument_token'),
                        'exchange': pos.get('exchange'),
                        'product': pos.get('product'),
                        'quantity': pos.get('quantity'),
                        'buy_qty': pos.get('buy_quantity'),
                        'sell_qty': pos.get('sell_quantity'),
                        'buy_price': pos.get('buy_price'),
                        'sell_price': pos.get('sell_price'),
                        'average_price': pos.get('average_price'),
                        'last_price': pos.get('last_price'),
                        'pnl': pos.get('pnl'),
                        'm2m': pos.get('m2m'),
                        'realised': pos.get('realised'),
                        'unrealised': pos.get('unrealised'),
                        'value': pos.get('value')
                    })
            
        if all_positions:
            cursor.executemany("""
                    INSERT INTO my_schema.positions (
                        position_type, trading_symbol, instrument_token,
                        exchange, product, quantity, buy_qty, sell_qty, buy_price,
                        sell_price, average_price, last_price, pnl, m2m, realised,
                        unrealised, value
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (position_type, instrument_token) DO UPDATE
                    SET trading_symbol = EXCLUDED.trading_symbol,
                        exchange = EXCLUDED.exchange,
                        product = EXCLUDED.product,
                        quantity = EXCLUDED.quantity,
                        buy_qty = EXCLUDED.buy_qty,
                        sell_qty = EXCLUDED.sell_qty,
                        buy_price = EXCLUDED.buy_price,
                        sell_price = EXCLUDED.sell_price,
                        average_price = EXCLUDED.average_price,
                        last_price = EXCLUDED.last_price,
                        pnl = EXCLUDED.pnl,
                        m2m = EXCLUDED.m2m,
                        realised = EXCLUDED.realised,
                        unrealised = EXCLUDED.unrealised,
                        value = EXCLUDED.value
                """, [
                    (
                        pos['position_type'], pos['trading_symbol'],
                        pos['instrument_token'], pos['exchange'], pos['product'],
                        pos['quantity'], pos['buy_qty'], pos['sell_qty'],
                        pos['buy_price'], pos['sell_price'], pos['average_price'],
                        pos['last_price'], pos['pnl'], pos['m2m'],
                        pos['realised'], pos['unrealised'], pos['value']
                    ) for pos in all_positions
                ])
        conn.commit()
    


def fetch_and_save_holdings():
    print('fetching holdings')
    holdings = kite.holdings()  # save into postgres table with current date
    #logging.info(f"User holdings: {holdings}")
    #print(holdings)
    print(type(holdings))

    try:
        if not holdings:
            logging.warning("No holdings found")
            return []
    except Exception as e:
        logging.error(f"Error XXXX: {e}")

    print('validating holdings')
    valid_holdings = []
    try:
        print('validating holdings')
        for h in holdings:
            # Validate critical fields
            if not h.get('instrument_token'):
                logging.warning(f"Skipping holding with missing instrument_token: {h}")
                continue
            holding = {
                'tradingsymbol': h.get('tradingsymbol', ''),
                'instrument_token': h.get('instrument_token', 0),
                'exchange': h.get('exchange', ''),
                'isin': h.get('isin', ''),
                'quantity': h.get('quantity', 0),
                't1_quantity': h.get('t1_quantity', 0),
                'authorised_quantity': h.get('authorised_quantity', 0),
                'average_price': h.get('average_price', 0.0),
                'close_price': h.get('close_price', None),
                'last_price': h.get('last_price', None),
                'pnl': h.get('pnl', None),
                'collateral_quantity': h.get('collateral_quantity', 0),
                'collateral_type': h.get('collateral_type', None)
            }
            valid_holdings.append(holding)
            logging.debug(f"Processed holding: {holding}")
    except Exception as e:
        logging.error(f"Error in validating: {e}")
       
    try:
        print('saving holdings')
        if valid_holdings:
            cursor.executemany("""
                INSERT INTO my_schema.holdings (
                    trading_symbol, instrument_token, exchange,
                    isin, quantity, t1_quantity, authorised_quantity, average_price,
                    close_price, last_price, pnl, collateral_quantity, collateral_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fetch_timestamp, instrument_token) DO UPDATE
                SET trading_symbol = EXCLUDED.trading_symbol,
                    exchange = EXCLUDED.exchange,
                    isin = EXCLUDED.isin,
                    quantity = EXCLUDED.quantity,
                    t1_quantity = EXCLUDED.t1_quantity,
                    authorised_quantity = EXCLUDED.authorised_quantity,
                    average_price = EXCLUDED.average_price,
                    close_price = EXCLUDED.close_price,
                    last_price = EXCLUDED.last_price,
                    pnl = EXCLUDED.pnl,
                    collateral_quantity = EXCLUDED.collateral_quantity,
                    collateral_type = EXCLUDED.collateral_type
            """, [
                (
                    h['tradingsymbol'], h['instrument_token'],
                    h['exchange'], h['isin'], h['quantity'],
                    h['t1_quantity'], h['authorised_quantity'],
                    h['average_price'], h['close_price'], h['last_price'],
                    h['pnl'], h['collateral_quantity'], h['collateral_type']
                ) for h in valid_holdings
            ])
    except Exception as e:
        logging.error(f"Error here: {e}")
    
    conn.commit()


def fetch_and_save_trades():
    print('fetching trades')
    trades = kite.trades()  # save into postgres table with current date
    logging.info(f"User trades: {trades}")
    #print(trades)
    if not trades:
        logging.warning("No trades found")
        return []
    cursor.executemany("""
                INSERT INTO my_schema.trades (
                    trade_id, order_id, exchange_order_id, exchange,
                    trading_symbol, instrument_token, transaction_type, quantity,
                    average_price, trade_timestamp, exchange_timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trade_id) DO UPDATE
                SET order_id = EXCLUDED.order_id,
                    exchange_order_id = EXCLUDED.exchange_order_id,
                    exchange = EXCLUDED.exchange,
                    trading_symbol = EXCLUDED.trading_symbol,
                    instrument_token = EXCLUDED.instrument_token,
                    transaction_type = EXCLUDED.transaction_type,
                    quantity = EXCLUDED.quantity,
                    average_price = EXCLUDED.average_price,
                    trade_timestamp = EXCLUDED.trade_timestamp,
                    exchange_timestamp = EXCLUDED.exchange_timestamp
            """, [
                (
                    trade.get('trade_id'), trade.get('order_id'),
                    trade.get('exchange_order_id'), trade.get('exchange'),
                    trade.get('trading_symbol'), trade.get('instrument_token'),
                    trade.get('transaction_type'), trade.get('quantity'),
                    trade.get('average_price'),
                    pd.to_datetime(trade.get('trade_timestamp')) if trade.get('trade_timestamp') else None,
                    pd.to_datetime(trade.get('exchange_timestamp')) if trade.get('exchange_timestamp') else None
                ) for trade in trades
            ])
    conn.commit()


def fetch_and_save_margins():
    print('fetching margins')
    margins = kite.margins()    # save into postgres table with current date
    logging.info(f"User margins: {margins}")
    #print(margins)
    if not margins:
        logging.warning("No margins data found")
        return []
            
    try:
        valid_margins = []
        for margin_type in ['equity', 'commodity']:
            margin = margins.get(margin_type, {})
            if not margin:
                logging.warning(f"No data for margin_type: {margin_type}")
                continue
            
            valid_margins.append({
                'margin_type': margin_type,
                'enabled': margin.get('enabled', False),
                'net': margin.get('net', 0.0),
                'available_adhoc_margin': margin.get('available', {}).get('adhoc_margin', 0.0),
                'available_cash': margin.get('available', {}).get('cash', 0.0),
                'available_opening_balance': margin.get('available', {}).get('opening_balance', 0.0),
                'available_live_balance': margin.get('available', {}).get('live_balance', 0.0),
                'available_collateral': margin.get('available', {}).get('collateral', 0.0),
                'available_intraday_payin': margin.get('available', {}).get('intraday_payin', 0.0),
                'utilised_debits': margin.get('utilised', {}).get('debits', 0.0),
                'utilised_exposure': margin.get('utilised', {}).get('exposure', 0.0),
                'utilised_m2m_realised': margin.get('utilised', {}).get('m2m_realised', 0.0),
                'utilised_m2m_unrealised': margin.get('utilised', {}).get('m2m_unrealised', 0.0),
                'utilised_option_premium': margin.get('utilised', {}).get('option_premium', 0.0),
                'utilised_payout': margin.get('utilised', {}).get('payout', 0.0),
                'utilised_span': margin.get('utilised', {}).get('span', 0.0),
                'utilised_holding_sales': margin.get('utilised', {}).get('holding_sales', 0.0),
                'utilised_turnover': margin.get('utilised', {}).get('turnover', 0.0),
                'utilised_liquid_collateral': margin.get('utilised', {}).get('liquid_collateral', 0.0),
                'utilised_stock_collateral': margin.get('utilised', {}).get('stock_collateral', 0.0),
                'utilised_equity': margin.get('utilised', {}).get('equity', 0.0),
                'utilised_delivery': margin.get('utilised', {}).get('delivery', 0.0)
            })
            logging.debug(f"Processed margin: {valid_margins[-1]}")
                
        if valid_margins:
            cursor.executemany("""
                INSERT INTO my_schema.margins (
                    margin_type, enabled, net,
                    available_adhoc_margin, available_cash, available_opening_balance,
                    available_live_balance, available_collateral, available_intraday_payin,
                    utilised_debits, utilised_exposure, utilised_m2m_realised,
                    utilised_m2m_unrealised, utilised_option_premium, utilised_payout,
                    utilised_span, utilised_holding_sales, utilised_turnover,
                    utilised_liquid_collateral, utilised_stock_collateral,
                    utilised_equity, utilised_delivery
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fetch_timestamp, margin_type) DO UPDATE
                SET enabled = EXCLUDED.enabled,
                    net = EXCLUDED.net,
                    available_adhoc_margin = EXCLUDED.available_adhoc_margin,
                    available_cash = EXCLUDED.available_cash,
                    available_opening_balance = EXCLUDED.available_opening_balance,
                    available_live_balance = EXCLUDED.available_live_balance,
                    available_collateral = EXCLUDED.available_collateral,
                    available_intraday_payin = EXCLUDED.available_intraday_payin,
                    utilised_debits = EXCLUDED.utilised_debits,
                    utilised_exposure = EXCLUDED.utilised_exposure,
                    utilised_m2m_realised = EXCLUDED.utilised_m2m_realised,
                    utilised_m2m_unrealised = EXCLUDED.utilised_m2m_unrealised,
                    utilised_option_premium = EXCLUDED.utilised_option_premium,
                    utilised_payout = EXCLUDED.utilised_payout,
                    utilised_span = EXCLUDED.utilised_span,
                    utilised_holding_sales = EXCLUDED.utilised_holding_sales,
                    utilised_turnover = EXCLUDED.utilised_turnover,
                    utilised_liquid_collateral = EXCLUDED.utilised_liquid_collateral,
                    utilised_stock_collateral = EXCLUDED.utilised_stock_collateral,
                    utilised_equity = EXCLUDED.utilised_equity,
                    utilised_delivery = EXCLUDED.utilised_delivery
            """, [
                (
                    m['margin_type'], m['enabled'], m['net'],
                    m['available_adhoc_margin'], m['available_cash'], m['available_opening_balance'],
                    m['available_live_balance'], m['available_collateral'], m['available_intraday_payin'],
                    m['utilised_debits'], m['utilised_exposure'], m['utilised_m2m_realised'],
                    m['utilised_m2m_unrealised'], m['utilised_option_premium'], m['utilised_payout'],
                    m['utilised_span'], m['utilised_holding_sales'], m['utilised_turnover'],
                    m['utilised_liquid_collateral'], m['utilised_stock_collateral'],
                    m['utilised_equity'], m['utilised_delivery']
                ) for m in valid_margins
            ])
            conn.commit()
            logging.info(f"Stored {len(valid_margins)} margins")
        return valid_margins
    except Exception as e:
        logging.error(f"Error fetching/storing margins: {e}")
        return []


# Function to generate a new access token
def generate_new_access_token(request_token):
    print('generate token')
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        redis_client.set("kite_access_token", request_token)
        redis_client.set("kite_access_token_timestamp", str(time.time()))
                         
        logging.info(f"New access token generated: {access_token}")

        try:
            init_postgres_conn()
            print('1')
            fetch_and_save_holdings()
            print('2')
            fetch_and_save_orders()
            print('3')
            fetch_and_save_positions()
            print('4')
            fetch_and_save_trades()
            print('5')
            fetch_and_save_margins()
            print('6')
            fetch_and_save_profile()
        except Exception as e:
            logging.error(f"Error saving data to database: {e}")

        return access_token
    except Exception as e:
        logging.error(f"Failed to generate new access token: {e}")
        return None


# FastAPI app for capturing request_token
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def home(
    action: str = Query(None),
    type: str = Query(None),
    status: str = Query(None),
    request_token: str = Query(None)
):
    kite_access_token_timestamp = redis_client.get("kite_access_token_timestamp")
    if request_token and status == "success" and action == "login" and type == "login":
        # Handle redirect from Zerodha with request_token
        redis_client.set("kite_request_token", request_token)
        logging.info(f"Received request_token: {request_token}")
        token_access = request_token
        return f"""
            <h1>Kite Connect Authentication</h1>
            <p>Request token received: {request_token}</p>
            <p>Request token saved at timestamp: {kite_access_token_timestamp}</p>
            <p>Access token generation in progress...</p>
            <p>Saving Tick data to Database...</p>
        """
    else:
        # Show login page
        login_url = kite.login_url()
        return f"""
            <h1>Kite Connect Authentication</h1>
            <p>Request token saved at timestamp: {kite_access_token_timestamp}</p>
            <p><a href="{login_url}">Click here to log in to Kite Connect</a></p>
            <p>Please authenticate to generate a new access token.</p>
        """


@app.get("/redirect", response_class=HTMLResponse)
async def handle_redirect(request_token: str = None):
    print('redirects')
    if not request_token:
        raise HTTPException(status_code=400, detail="No request_token provided")
    # Store request_token in Redis
    redis_client.set("kite_request_token", request_token)
    return f"""
        <h1>Kite Connect Authentication</h1>
        <p>Request token received: {request_token}</p>
        <p>Access token generation in progress...</p>
    """

# To fetch TPO plots from http://localhost:8000/tpo_plot/premarket

@app.get("/tpo_plot/{session_type}", response_class=FileResponse)
async def get_tpo_plot(session_type: str):
    if session_type not in ["premarket", "regular"]:
        raise HTTPException(status_code=400, detail="Invalid session type")
    plot_path = f"/app/tpo_{session_type}.png"
    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type="image/png", filename=f"tpo_{session_type}.png")
    raise HTTPException(status_code=404, detail="Plot not found")


# Main logic
def get_access_token():
    print('get access token')
    global ACCESS_TOKEN
    ACCESS_TOKEN = ''
    # Check if existing access token is provided and valid
    if len(ACCESS_TOKEN) > 0 and is_access_token_valid(ACCESS_TOKEN):
        return ACCESS_TOKEN
    else:
        # Clear any existing request_token in Redis
        redis_client.delete("kite_request_token")
        print('2')
        
        # Log login URL
        login_url = kite.login_url()
        
        logging.info(f"Login URL: {login_url}")
        #logging.info("Please open the login URL in a browser to authenticate (http://localhost:8000).") # why?

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
        new_access_token = generate_new_access_token(request_token)
        if new_access_token:
            ACCESS_TOKEN = new_access_token
            return new_access_token
        else:
            raise Exception("Could not obtain a valid access token")


