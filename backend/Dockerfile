# Multi-stage production Dockerfile for MedScribe AI FastAPI backend
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for compiled wheels (e.g. webrtcvad)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Install runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Configurable Whisper model with base.en default for low-memory environments (Render Free tier)
ARG WHISPER_MODEL=base.en
ENV WHISPER_MODEL=${WHISPER_MODEL}
ENV WHISPER_CPU_THREADS=1

# Pre-download the configured faster-whisper model at Docker build time
# This avoids download timeouts and keeps startup memory & time low
RUN python3 -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.getenv('WHISPER_MODEL', 'base.en'), device='cpu', compute_type='int8')"

# Copy backend application source
COPY . .

# Set default server environment
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Start Uvicorn bound to 0.0.0.0 and dynamic PORT with exactly 1 worker to keep RAM under 512MB
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
