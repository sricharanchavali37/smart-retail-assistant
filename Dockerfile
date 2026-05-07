# Smart Retail Assistant — Dockerfile
# Python 3.10 slim base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY backend/ ./backend/
COPY data/raw/ ./data/raw/
COPY scripts/ ./scripts/
COPY notebooks/ ./notebooks/
COPY adf/ ./adf/


# Run Phase 1 to generate ML models and processed data
RUN python backend/ml/run_phase1.py

# Build FAISS index for RAG
RUN python backend/agents/build_index.py

# Expose port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "9000"]