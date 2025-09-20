from fastapi import FastAPI
import psycopg2
from psycopg2.extras import RealDictCursor
import redis

app = FastAPI()

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="postgres",
        database="mydb",
        user="postgres",
        password="postgres"
    )

def get_redis_connection():
    return redis.Redis(host="redis", port=6379, db=0, decode_responses=True)

@app.get("/users")
def get_users():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM my_schema.users")
            users = cur.fetchall()
        return users
    finally:
        conn.close()

@app.get("/health")
def health_check():
    return {"status": "healthy"}
    
@app.get("/redis_health")
def health_check_redis():
    try:
        redis_conn = get_redis_connection()
        redis_conn.ping()
        redis_status = "connected"
    except Exception:
        redis_status = "disconnected"
    
    return {"status": "healthy", "redis": redis_status}
    

        
