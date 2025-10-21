from Boilerplate import *

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

@app.get("/", response_class=HTMLResponse)
async def home(
    action: str = Query(None),
    type: str = Query(None),
    status: str = Query(None),
    request_token: str = Query(None)
):
    #await get_access_token()
    global token_access
    
    kite_access_token_timestamp = redis_client.get("kite_access_token_timestamp")
    if request_token and status == "success" and action == "login" and type == "login":
        # Handle redirect from Zerodha with request_token
        redis_client.set("kite_request_token", request_token)
        logging.info(f"Received request_token: {request_token}")
        token_access = request_token

        # Trigger get_access_token to generate and save access token
        access_token = get_access_token()  # Call your function here

        return f"""
            <h1>Kite Connect Authentication</h1>
            <p>Request token received: {request_token}</p>
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
        #request_token = redis_client.get("kite_request_token")
        #get_access_token()
        #printrequest_token)
        return f"""
            <h1>Kite Connect Authentication</h1>
            <p><a href="{login_url}">Click here to log in to Kite Connect</a></p>
            <p>Please authenticate to generate a new access token.</p>
        """


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


