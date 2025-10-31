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

# Create logs directory for cron jobs
RUN mkdir -p /app/logs && chmod 755 /app/logs

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000

# Copy crontab file to the cron.d directory and give it the right permissions.
# The cron daemon will pick it up automatically.
COPY app/crontab.txt /etc/cron.d/crontab
RUN chmod 0644 /etc/cron.d/crontab && \
    # Ensure crontab file ends with newline (cron requirement)
    echo "" >> /etc/cron.d/crontab

# Default command (can be overridden in docker-compose.yml)
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["python", "MarketProf1.py"]
