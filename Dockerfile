FROM python:3.10-slim

# Install system build dependencies and graphics libraries for RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libxrender1 \
    libgl1 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install PyTorch CPU version first to minimize image size
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and files needed at runtime
COPY src/ ./src
COPY configs/ ./configs
COPY model.pt .
COPY graphmol.db .

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port
EXPOSE 8000

# Start FastAPI using the dynamic PORT env variable set by Render
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT"]
