# Use lightweight Python base
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for safety
RUN useradd -m appuser

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Hugging Face cache (for embeddings/models)
RUN mkdir -p /app/huggingface_cache && chmod -R 777 /app/huggingface_cache
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (7860 for Hugging Face Spaces, local testing)
EXPOSE 7860

# Set Hugging Face cache env
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
