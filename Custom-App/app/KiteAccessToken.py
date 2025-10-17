import os
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from kiteconnect import KiteConnect
import redis
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)

#global ACCESS_TOKEN
#ACCESS_TOKEN = None

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

# Function to validate access token
def is_access_token_valid(access_token):
    print('cheking token')
    try:
        kite.set_access_token(access_token) # Set the access token
        kite.margins()  # Make a simple API call to validate (e.g., get margins)
        logging.info("Existing access token is valid")
        return True
    except Exception as e:
        logging.error(f"Access token validation failed: {e}")
        return False

# Function to generate a new access token
def generate_new_access_token(request_token):
    print('generate token')
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
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
    if request_token and status == "success" and action == "login" and type == "login":
        # Handle redirect from Zerodha with request_token
        redis_client.set("kite_request_token", request_token)
        logging.info(f"Received request_token: {request_token}")
        token_access = request_token
        return f"""
            <h1>Kite Connect Authentication</h1>
            <p>Request token received: {request_token}</p>
            <p>Access token generation in progress...</p>
        """
    else:
        # Show login page
        login_url = kite.login_url()
        return f"""
            <h1>Kite Connect Authentication</h1>
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


""" def get_access_token():
    # Check if existing access token is provided and valid
    if ACCESS_TOKEN and is_access_token_valid(ACCESS_TOKEN):
        return ACCESS_TOKEN
    else:
        # Start FastAPI server in a separate thread
        server_thread = Thread(target=start_fastapi_server, daemon=True)
        server_thread.start()

        # Log login URL and wait for request_token
        login_url = kite.login_url()
        logging.info(f"Login URL: {login_url}")
        logging.info("Please open the login URL in a browser to authenticate.")

        # Wait for request_token from /redirect endpoint
        request_token_event.wait(timeout=300)  # Wait up to 5 minutes
        if not request_token:
            raise Exception("Failed to receive request_token within timeout")

        # Generate new access token
        new_access_token = generate_new_access_token(request_token)
        if new_access_token:
            global ACCESS_TOKEN
            ACCESS_TOKEN = new_access_token  # Update global ACCESS_TOKEN
            return new_access_token
        else:
            raise Exception("Could not obtain a valid access token")
 """



""" if __name__ == "__main__":
    # Replace with your existing access token (if available, e.g., from a file or previous session)
    EXISTING_ACCESS_TOKEN = None  # Set to your current access token or None if not available

    try:
        ACCESS_TOKEN = get_access_token(EXISTING_ACCESS_TOKEN)
        print(f"Access Token: {ACCESS_TOKEN}")
    except Exception as e:
        logging.error(f"Error: {e}") """