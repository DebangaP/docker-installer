from kiteconnect import KiteConnect
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    API_KEY = os.environ["KITE_API_KEY"]
    API_SECRET = os.environ["KITE_API_SECRET"]
    print(f"API_KEY: {API_KEY}")
    print(f"API_SECRET: {API_SECRET}")
except KeyError as e:
    logging.error(f"Environment variable {e} not set")
    raise

# Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)

# Function to validate access token
def is_access_token_valid(access_token):
    try:
        # Set the access token
        kite.set_access_token(access_token)
        # Make a simple API call to validate (e.g., get margins)
        kite.margins()
        logging.info("Existing access token is valid")
        return True
    except Exception as e:
        logging.error(f"Access token validation failed: {e}")
        return False

# Function to generate a new access token
def generate_new_access_token(request_token):
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        logging.info(f"New access token generated: {access_token}")
        return access_token
    except Exception as e:
        logging.error(f"Failed to generate new access token: {e}")
        return None

# Main logic
def get_access_token(existing_access_token=None):
    # Check if existing access token is provided and valid
    if existing_access_token and is_access_token_valid(existing_access_token):
        return existing_access_token
    else:
        # Step 1: Print login URL for manual authentication
        logging.info(f"Login URL: {kite.login_url()}")
        # Step 2: Prompt user to log in and provide request_token
        request_token = input("Enter the request_token from the redirect URL: ")
        # Step 3: Generate new access token
        new_access_token = generate_new_access_token(request_token)
        if new_access_token:
            return new_access_token
        else:
            raise Exception("Could not obtain a valid access token")

if __name__ == "__main__":
    # Replace with your existing access token (if available, e.g., from a file or previous session)
    EXISTING_ACCESS_TOKEN = None  # Set to your current access token or None if not available

    try:
        ACCESS_TOKEN = get_access_token(EXISTING_ACCESS_TOKEN)
        print(f"Access Token: {ACCESS_TOKEN}")
    except Exception as e:
        logging.error(f"Error: {e}")