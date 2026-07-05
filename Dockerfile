# ─────────────────────────────────────────────────────────────────────────────
# CoreLLM – Dockerfile
#
# Strategy: models are downloaded at BUILD TIME and baked into the image.
# At runtime, Ollama just loads from disk — no internet needed, instant start.
#
# To change which models are available, update ALLOWED_MODELS and rebuild.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System deps + Ollama binary (runs as root during build) ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends curl zstd && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Create non-root user + give it a writable Ollama home ────────────────────
RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/.ollama && \
    chown -R user:user /home/user/.ollama

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    # Ollama stores models here (user-owned, not /root)
    OLLAMA_HOME=/home/user/.ollama \
    OLLAMA_MODELS=/home/user/.ollama/models \
    # ── IMPORTANT ────────────────────────────────────────────────────────────
    # HF Space secrets/variables are NOT available at build time.
    # Models must be listed HERE so they are downloaded during the Docker build.
    # This value is also used as the runtime allowlist.
    # To add/remove models: edit this line, then push to trigger a rebuild.
    ALLOWED_MODELS="lfm2.5-thinking:1.2b" \
    # Optional Bearer token — override this via HF Space Secret at runtime
    API_KEY="CHANGE_ME"

# ── Pre-download all ALLOWED_MODELS at build time ────────────────────────────
# Start Ollama in the background, pull every model, then shut it down.
# Models end up in /home/user/.ollama/models and are baked into the layer.
RUN ollama serve & \
    SERVE_PID=$! && \
    echo "Waiting for Ollama to be ready..." && \
    for i in $(seq 1 30); do \
      curl -sf http://127.0.0.1:11434 > /dev/null && break; \
      sleep 1; \
    done && \
    echo "Pulling models: $ALLOWED_MODELS" && \
    echo "$ALLOWED_MODELS" | tr ',' '\n' | while IFS= read -r m; do \
      m=$(echo "$m" | tr -d '[:space:]'); \
      if [ -n "$m" ]; then \
        echo "  → Pulling $m ..."; \
        ollama pull "$m"; \
        echo "  ✓ $m done"; \
      fi; \
    done && \
    kill $SERVE_PID && \
    echo "All models cached."

WORKDIR $HOME/app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY --chown=user . .
RUN chmod +x start.sh

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["./start.sh"]