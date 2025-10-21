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
import asyncio
import subprocess
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
kite.set_access_token(redis_client.get("kite_access_token"))

def get_db_connection():
    return psycopg2.connect(
        host="postgres",
        database="mydb",
        user="postgres",
        password="postgres"
    )

conn = get_db_connection()
cursor = conn.cursor()