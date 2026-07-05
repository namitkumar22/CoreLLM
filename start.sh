#!/bin/bash
set -e

# ── 1. Start Ollama (models are already on disk — no pulling needed) ─────────
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# ── 2. Wait until Ollama is ready ────────────────────────────────────────────
echo "Waiting for Ollama to become ready..."
MAX_WAIT=30
COUNT=0
until curl -sf http://127.0.0.1:11434 > /dev/null; do
  sleep 1
  COUNT=$((COUNT+1))
  if [ "$COUNT" -ge "$MAX_WAIT" ]; then
    echo "ERROR: Ollama did not start within ${MAX_WAIT}s" >&2
    exit 1
  fi
done
echo "✓ Ollama is ready. Models on disk:"
ollama list

# ── 3. Start the FastAPI gateway ─────────────────────────────────────────────
echo "Starting CoreLLM API gateway on port 7860..."
exec uvicorn app:app --host 0.0.0.0 --port 7860