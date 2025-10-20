import os
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException   
import redis
import time
from dotenv import load_dotenv
from psycopg2.extras import execute_batch
import psycopg2
import pandas as pd
from datetime import datetime

#import InsertOHLC   # run this code to insert latest data

# Configure logging
logging.basicConfig(level=logging.INFO)

token_access = ''

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

# Import functions at module level
import KiteFetchData

def get_db_connection():
    return psycopg2.connect(
        host="postgres",
        database="mydb",
        user="postgres",
        password="postgres"
    )

conn = get_db_connection()
cursor = conn.cursor()

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
        logging.info(f"New access token generated: {access_token}")

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
             <p>Access token saved at timestamp: {datetime.fromtimestamp(float(kite_access_token_timestamp))} UTC (+5:30 hrs)</p>
            <p>Saving Tick data to Database...</p>
            <a href="http://localhost:3001/d/my-dashboard/sample-dashboard?orgId=1&from=now-90d&to=now"> GO TO My Dashboard</a>
            <br>
            <hr>
            <a href="http://127.0.0.1:8000/fetch_data"> Refresh Data </a>
            <br>
            
        """
    else:
        # Show login page
        login_url = kite.login_url()
        return f"""
            <h1>Kite Connect Authentication</h1>
            <p>Access token saved at timestamp: {datetime.fromtimestamp(float(kite_access_token_timestamp))}</p>
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


@app.get("/fetch_data", response_class=HTMLResponse)
def fetch_data(request_token: str = None):
    print('Fetching Data')

    kite.set_access_token(redis_client.get("kite_access_token"))

    KiteFetchData.fetch_and_save_holdings(kite, cursor)
    KiteFetchData.fetch_and_save_orders(kite, cursor)
    KiteFetchData.fetch_and_save_positions(kite, cursor)
    KiteFetchData.fetch_and_save_trades(kite, cursor)
    KiteFetchData.fetch_and_save_margins(kite, cursor)
    
    return f"""
        <h1>Refreshing Data</h1>
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
# Note here that if a valid access token is available in Redis then login is not necessary and the all the code would run successfully
#
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


