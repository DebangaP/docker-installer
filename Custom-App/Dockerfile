FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt kiteconnect openpyxl reportlab

# Copy application code
COPY app/ .

ENV TZ=Asia/Kolkata
# Install cron and set timezone
RUN apt-get update && apt-get install -y cron tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000

# Apply cron job
RUN crontab /app/crontab.txt
# Start cron in foreground
CMD ["cron", "-f"]

# Default command (can be overridden in docker-compose.yml)
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

#CMD ["python", "MarketProf1.py"]

