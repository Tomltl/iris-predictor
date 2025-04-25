# Base image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary directories
COPY api/ api/
COPY app/ app/
COPY templates/ templates/
COPY mlruns/ mlruns/

# Expose Flask port
EXPOSE 8000

# Run Flask application
CMD ["python", "api/main.py"]
