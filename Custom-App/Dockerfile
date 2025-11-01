FROM python:3.10-slim

WORKDIR /app

# Install build dependencies for cmdstan (needed for Prophet)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt kiteconnect openpyxl reportlab

# Install cmdstan (required for Prophet)
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()" || \
    echo "Warning: cmdstan installation may have failed, but continuing..."

# Remove build dependencies to reduce image size (optional - uncomment if you want smaller image)
# RUN apt-get purge -y build-essential make gcc g++ && \
#     apt-get autoremove -y && \
#     apt-get clean

# Copy application code
COPY app/ .

ENV TZ=Asia/Kolkata
# Install cron and set timezone
RUN apt-get update && apt-get install -y --no-install-recommends cron tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

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
