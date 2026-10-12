FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLM_PROVIDER=none \
    PORT=8001 \
    HF_HOME=/app/models

# Set work directory
WORKDIR /app

# Install system dependencies (build-essential helps with some binary wheel compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the application code
COPY backend/ /app/backend/
COPY scripts/ /app/scripts/

# Ensure data directory exists
RUN mkdir -p /app/backend/data

# Pre-download and verify pinned ONNX embedding model during build
# This avoids downloading ~90MB on the first public request or failing due to egress limits.
RUN python /app/scripts/download_onnx_model.py && \
    python -c "import sys; sys.path.insert(0, '/app/backend'); from app.rag.embeddings import OnnxEmbeddingModel; m = OnnxEmbeddingModel(); m.dimension"

# Initialize SQLite database and ChromaDB index with seed data
# This creates a baseline database baked into the image.
RUN python /app/scripts/setup_data.py && \
    python /app/scripts/ingest_labels.py --seed && \
    python /app/scripts/build_vector_index.py

# Set offline mode and CPU thread limits for runtime execution on Render
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Switch work directory for uvicorn
WORKDIR /app/backend

# Start the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
