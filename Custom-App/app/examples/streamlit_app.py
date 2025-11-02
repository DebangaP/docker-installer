import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

st.title("PostgreSQL Data Dashboard")

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="postgres",
        database="mydb",
        user="youruser",
        password="yourpassword"
    )

# Fetch and display data
conn = get_db_connection()
try:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM my_schema.users")
        users = cur.fetchall()
    df = pd.DataFrame(users)
    st.write("Users Table")
    st.dataframe(df)
finally:
    conn.close()