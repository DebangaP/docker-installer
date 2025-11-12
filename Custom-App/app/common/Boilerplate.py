import os
import logging
import traceback
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
from datetime import datetime, date
from kiteconnect import KiteTicker

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Redis client
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
access_token = redis_client.get("kite_access_token")
logging.info(f"Access Token: {access_token}")

load_dotenv()

try:
    API_KEY = os.getenv("KITE_API_KEY")
    API_SECRET = os.getenv("KITE_API_SECRET")
except KeyError as e:
    logging.error(f"Environment variable {e} not set")

logging.info(f"API_KEY: {API_KEY}")
logging.info(f"API_SECRET: {API_SECRET}")

# Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)

# Get access token from Redis
access_token = redis_client.get("kite_access_token")
if access_token:
    # Handle both string and bytes from Redis
    if isinstance(access_token, bytes):
        access_token = access_token.decode('utf-8')
    kite.set_access_token(access_token)
    logging.info("Access token set from Redis")
else:
    logging.warning("No access token found in Redis")

def get_db_connection():
    return psycopg2.connect(
        host="postgres",
        database="mydb",
        user="postgres",
        password="postgres"
    )

def get_access_token():
    """
    Get access token from Redis
    Returns the access token string or None if not found
    """
    token = redis_client.get("kite_access_token")
    if token:
        # Handle both string and bytes from Redis
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        return token
    return None

def log_stock_price_fetch_error(scrip_id: str, error: Exception, function_name: str = None):
    """
    Log stock price fetch errors to the database
    
    Args:
        scrip_id: Stock symbol/identifier
        error: Exception object that occurred
        function_name: Name of the function where error occurred
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Truncate long messages to fit database constraints
        if len(error_message) > 10000:
            error_message = error_message[:10000]
        if len(stack_trace) > 50000:
            stack_trace = stack_trace[:50000]
        if function_name and len(function_name) > 200:
            function_name = function_name[:200]
        
        cursor.execute("""
            INSERT INTO my_schema.stock_price_fetch_errors 
            (scrip_id, error_type, error_message, function_name, stack_trace, fetch_date)
            VALUES (%s, %s, %s, %s, %s, CURRENT_DATE)
        """, (scrip_id, error_type, error_message, function_name, stack_trace))
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        # If logging itself fails, just log to standard logger
        logging.error(f"Failed to log price fetch error to database: {e}")
        logging.error(f"Original error for {scrip_id} in {function_name}: {error}")

conn = get_db_connection()
cursor = conn.cursor()