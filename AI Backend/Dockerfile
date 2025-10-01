# Use lightweight Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create Hugging Face cache dir and give full permissions
RUN mkdir -p /app/huggingface_cache && chmod -R 777 /app/huggingface_cache

# Change ownership of /app to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Hugging Face Spaces allowed port
EXPOSE 7860

# Hugging Face cache environment variables
ENV HF_HOME=/app/huggingface_cache
ENV HF_DATASETS_CACHE=/app/huggingface_cache
ENV TORCH_HOME=/app/huggingface_cache


# Start FastAPI app on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
