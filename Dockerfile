# Aether Edge AI Engine - Production Dockerfile
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    AETHER_LLM_ALLOW_REMOTE=true

# Install OS libraries for OpenCV, audio, and camera device support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libasound2-dev \
    portaudio19-dev \
    espeak \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -U pip setuptools \
    && pip install --no-cache-dir -e ".[ci,server]" httpx

# Copy project source
COPY . .

EXPOSE 8000

# Default command: Headless FastAPI & WebSocket Server + Web Admin Portal
CMD ["python", "main.py", "--server", "--host", "0.0.0.0", "--port", "8000"]
