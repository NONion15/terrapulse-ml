FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies (libgomp1 for OpenMP acceleration in LightGBM/CatBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ ./app/
COPY model/ ./model/
COPY data/ ./data/

# Hugging Face default port
EXPOSE 7860

# Run with Gunicorn WSGI server
CMD exec gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-7860} app.server:app
