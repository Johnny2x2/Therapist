# RunPod serverless image for the Therapist engine.
# Runs Ollama + the streaming handler. Models and app data live on the mounted
# network volume (/runpod-volume) at runtime.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_MODELS=/runpod-volume/ollama/models \
    THERAPIST_DATA_DIR=/runpod-volume/therapist-data

# System deps: curl for healthchecks, ca-certificates for TLS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama.
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Python deps first for better layer caching.
COPY requirements-server.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-server.txt

# App code + prompts.
COPY src ./src
COPY prompts ./prompts
COPY scripts/serverless_entry.sh ./scripts/serverless_entry.sh
RUN chmod +x ./scripts/serverless_entry.sh

CMD ["./scripts/serverless_entry.sh"]
