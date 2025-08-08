# =====================
# Stage 1: Build Image
# =====================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY forex_pipeline.py .

# Create models directory for saved models
RUN mkdir -p models

# Expose API port
EXPOSE 8000

# Environment variables
ENV FASTFOREX_API_KEY=your_api_key_here

# Run the app with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "forex_pipeline:app"]
