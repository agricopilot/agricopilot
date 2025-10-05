# =============================================
# 🚀 AgriCopilot — Optimized Dockerfile (Hackathon Ready)
# =============================================

# --- Base Image ---
FROM python:3.10-slim

# --- System Setup ---
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# --- Create User ---
RUN useradd -m appuser
WORKDIR /app

# --- Copy Requirements First for Layer Caching ---
COPY requirements.txt .

# --- Install Core Python Dependencies ---
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
        langchain-huggingface \
        fastapi \
        uvicorn \
        pillow \
        transformers \
        accelerate \
        sentence-transformers \
        faiss-cpu \
        pandas

# --- Copy the Entire Application ---
COPY . .

# --- Hugging Face Cache ---
RUN mkdir -p /app/huggingface_cache && chmod -R 777 /app/huggingface_cache
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache
ENV HF_DATASETS_CACHE=/app/huggingface_cache

# --- Environment Variables ---
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# --- Security: Non-root user ---
RUN chown -R appuser:appuser /app
USER appuser

# --- Expose Port for Docker MCP Gateway / Hugging Face Spaces ---
EXPOSE 7860

# --- Run Server ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
