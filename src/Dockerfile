# Base image with Python and CUDA 12.1 runtime
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Install System Dependencies (ffmpeg is critical for VAD/Whisper/Audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ /app/app/

# Expose FastAPI port
EXPOSE 8000

# Entrypoint script or uvicorn server launch
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
