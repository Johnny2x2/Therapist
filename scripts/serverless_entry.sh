#!/usr/bin/env bash
# Entrypoint for the RunPod serverless worker.
# Boots Ollama against the network-volume model store, then starts the handler.
set -euo pipefail

# Network volume is mounted at /runpod-volume on serverless workers.
export OLLAMA_MODELS="${OLLAMA_MODELS:-/runpod-volume/ollama/models}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export THERAPIST_DATA_DIR="${THERAPIST_DATA_DIR:-/runpod-volume/therapist-data}"

mkdir -p "$OLLAMA_MODELS" "$THERAPIST_DATA_DIR"

echo "Starting ollama serve (models: $OLLAMA_MODELS)..."
ollama serve &

# Wait for the Ollama HTTP API to come up.
for i in $(seq 1 60); do
  if curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Ollama is ready."
    break
  fi
  sleep 1
done

# Optionally pull the therapist model if it is not already on the volume.
# Set THERAPIST_PULL_MODEL to the model tag to enable (e.g. on first run).
if [ -n "${THERAPIST_PULL_MODEL:-}" ]; then
  echo "Ensuring model present: $THERAPIST_PULL_MODEL"
  ollama pull "$THERAPIST_PULL_MODEL" || echo "pull failed; continuing"
fi

echo "Starting Therapist serverless handler..."
exec python -u -m src.serverless_handler
