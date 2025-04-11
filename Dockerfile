# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose port (FastAPI default)
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
