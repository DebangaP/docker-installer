FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt kiteconnect

# Copy application code
COPY app/ .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command (can be overridden in docker-compose.yml)
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["python", "MarketProf1.py"]

