from fastapi import FastAPI
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI()

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="postgres",
        database="mydb",
        user="youruser",
        password="yourpassword"
    )

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