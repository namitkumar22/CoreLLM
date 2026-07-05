"""
CoreLLM – FastAPI gateway wrapping Ollama on Hugging Face Spaces.

Features
--------
* Models are pre-baked into the Docker image at build time.
* ALLOWED_MODELS env var is the allowlist — only these can be used.
* Auto-switching: any inference request for a different model automatically
  unloads the current model and loads the requested one. No manual /switch needed.
* /api/switch  → explicit graceful model swap.
* /api/load    → explicitly load a model into memory.
* /api/unload  → release a model from memory (stays on disk).
* /v1/chat/completions  → OpenAI-compatible endpoint.
* /api/generate, /api/chat  → raw Ollama endpoints.
* Optional Bearer-token auth via API_KEY env var.
* Structured logging throughout — every request, switch, and error is visible.
"""

import os
import time
import asyncio
import logging
import httpx
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse
from typing import Optional

# ── Logging setup ─────────────────────────────────────────────────────────────

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  │  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger("corellm")

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # we log requests ourselves

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY       = os.environ.get("API_KEY", "")
OLLAMA_URL    = "http://127.0.0.1:11434"
_allowed_raw  = os.environ.get("ALLOWED_MODELS", "")
ALLOWED_MODELS: list[str] = (
    [m.strip() for m in _allowed_raw.split(",") if m.strip()]
    if _allowed_raw else []
)

# The model currently loaded in Ollama's memory
_active_model: Optional[str] = None

# Lock so concurrent requests don't race on model switching
_switch_lock = asyncio.Lock()

# ── Startup banner ────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("  CoreLLM API Gateway starting up")
log.info("=" * 60)
log.info(f"  Auth enabled   : {'YES' if API_KEY else 'NO (open access)'}")
log.info(f"  Allowed models : {ALLOWED_MODELS if ALLOWED_MODELS else 'ALL (no restriction)'}")
log.info(f"  Ollama URL     : {OLLAMA_URL}")
log.info(f"  Log level      : {LOG_LEVEL}")
log.info("=" * 60)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CoreLLM", version="1.0.0")


# ── Request logging middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request and its response time + status."""
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"

    log.info(f"→ {request.method} {request.url.path}  [from {client_ip}]")

    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        log.error(f"✗ {request.method} {request.url.path}  [{elapsed:.0f}ms]  UNHANDLED ERROR: {exc}")
        raise

    elapsed = (time.perf_counter() - start) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    log.log(
        level,
        f"← {request.method} {request.url.path}  "
        f"[{response.status_code}]  {elapsed:.0f}ms",
    )
    return response


# ── Auth & validation helpers ─────────────────────────────────────────────────

def _check_auth(request: Request):
    """Raise 401 if API_KEY is set and the Authorization header doesn't match."""
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {API_KEY}":
        client_ip = request.client.host if request.client else "unknown"
        log.warning(f"AUTH FAILED — invalid or missing key from {client_ip}")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _check_allowed(model: str):
    """Raise 403 if the model is not in the ALLOWED_MODELS list."""
    if ALLOWED_MODELS and model not in ALLOWED_MODELS:
        log.warning(f"BLOCKED model request: '{model}' not in allowed list {ALLOWED_MODELS}")
        raise HTTPException(
            status_code=403,
            detail=(
                f"Model '{model}' is not available. "
                f"Allowed models: {ALLOWED_MODELS}"
            ),
        )


# ── Core Ollama operations ────────────────────────────────────────────────────

async def _ollama_post(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{OLLAMA_URL}{path}", json=body)
        r.raise_for_status()
        return r.json()


async def _unload_from_memory(model: str):
    """
    Tell Ollama to evict a model from RAM (it stays on disk).
    Uses keep_alive=0 — the official Ollama way to release memory.
    """
    log.info(f"UNLOAD  '{model}' — releasing from RAM (stays on disk)...")
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
            )
        elapsed = (time.perf_counter() - t0) * 1000
        log.info(f"UNLOAD  '{model}' done  [{elapsed:.0f}ms]")
    except Exception as e:
        log.warning(f"UNLOAD  '{model}' failed (non-fatal): {e}")


async def _load_into_memory(model: str):
    """
    Pre-warm a model — loads it from disk into RAM.
    Uses keep_alive=-1 to keep it resident until explicitly unloaded.
    """
    log.info(f"LOAD    '{model}' — reading from disk into RAM...")
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": -1},
        )
    elapsed = (time.perf_counter() - t0) * 1000
    log.info(f"LOAD    '{model}' ready  [{elapsed:.0f}ms]")


async def _ensure_model(model: str):
    """
    Auto-switch: if the requested model differs from the active one,
    unload the current model and load the new one.
    Uses a lock to prevent race conditions from concurrent requests.
    """
    global _active_model

    if _active_model == model:
        log.debug(f"MODEL   '{model}' already active — no switch needed")
        return

    async with _switch_lock:
        # Re-check inside lock (another coroutine may have switched while we waited)
        if _active_model == model:
            log.debug(f"MODEL   '{model}' was loaded by a concurrent request — skipping")
            return

        previous = _active_model
        log.info(f"SWITCH  '{previous}' → '{model}'")
        t0 = time.perf_counter()

        if previous:
            await _unload_from_memory(previous)

        await _load_into_memory(model)
        _active_model = model

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(f"SWITCH  complete — active model is now '{model}'  [{elapsed:.0f}ms total]")


# ── Exception handler ──────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        log.warning(f"HTTP {exc.status_code} on {request.url.path} — {exc.detail}")
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    log.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Health endpoints ──────────────────────────────────────────────────────────

@app.get("/")
def health():
    log.debug("Health check requested")
    return {
        "status": "ok",
        "active_model": _active_model,
        "allowed_models": ALLOWED_MODELS or "all",
    }


@app.get("/ping")
def ping():
    return {"status": "alive"}


@app.get("/api/models")
async def list_models(request: Request):
    """List all models that are on disk and in the allowlist."""
    _check_auth(request)
    log.info("Fetching model list from Ollama...")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        data = r.json()
    all_local = [m["name"] for m in data.get("models", [])]
    visible = [m for m in all_local if not ALLOWED_MODELS or m in ALLOWED_MODELS]
    log.info(f"Models on disk: {all_local}  |  Visible (allowed): {visible}")
    return {
        "models": visible,
        "active": _active_model,
        "allowed": ALLOWED_MODELS or "all",
    }


# ── Explicit model lifecycle ──────────────────────────────────────────────────

@app.post("/api/load")
async def load_model(request: Request):
    """
    Explicitly load a model into memory.
    Body: { "model": "qwen2.5:1.5b" }
    """
    _check_auth(request)
    body = await request.json()
    model: str = body.get("model", "").strip()
    if not model:
        log.warning("POST /api/load called without 'model' field")
        raise HTTPException(status_code=400, detail="'model' field is required")
    _check_allowed(model)
    log.info(f"Explicit load requested for '{model}'")
    await _ensure_model(model)
    return {"status": "loaded", "model": model}


@app.post("/api/unload")
async def unload_model(request: Request):
    """
    Release a model from RAM. It stays on disk.
    Body: { "model": "qwen2.5:1.5b" }
    """
    global _active_model
    _check_auth(request)
    body = await request.json()
    model: str = body.get("model", "").strip()
    if not model:
        log.warning("POST /api/unload called without 'model' field")
        raise HTTPException(status_code=400, detail="'model' field is required")
    log.info(f"Explicit unload requested for '{model}'")
    await _unload_from_memory(model)
    if _active_model == model:
        _active_model = None
        log.info("Active model cleared")
    return {"status": "unloaded", "model": model}


@app.post("/api/switch")
async def switch_model(request: Request):
    """
    Explicitly switch the active model.
    Body: { "model": "llama3.2:3b" }
    """
    _check_auth(request)
    body = await request.json()
    new_model: str = body.get("model", "").strip()
    if not new_model:
        log.warning("POST /api/switch called without 'model' field")
        raise HTTPException(status_code=400, detail="'model' field is required")
    _check_allowed(new_model)
    previous = _active_model
    log.info(f"Explicit switch: '{previous}' → '{new_model}'")
    await _ensure_model(new_model)
    return {
        "status": "switched",
        "previous_model": previous,
        "active_model": new_model,
    }


# ── Inference helpers ─────────────────────────────────────────────────────────

def _resolve_model(body: dict) -> str:
    """Pick model from request body, falling back to the active model."""
    model = body.get("model", "").strip() or _active_model
    if not model:
        log.warning("Inference request with no model specified and no active model")
        raise HTTPException(
            status_code=400,
            detail=(
                "No model specified and no model is currently active. "
                "Pass 'model' in the request body or call /api/load first."
            ),
        )
    return model


# ── Inference endpoints ───────────────────────────────────────────────────────

@app.post("/api/generate")
async def generate(request: Request):
    _check_auth(request)
    body = await request.json()
    model = _resolve_model(body)
    _check_allowed(model)
    prompt_preview = str(body.get("prompt", ""))[:80].replace("\n", " ")
    log.info(f"GENERATE  model='{model}'  prompt='{prompt_preview}...'")
    await _ensure_model(model)
    body["model"] = model
    t0 = time.perf_counter()
    result = await _ollama_post("/api/generate", body)
    elapsed = (time.perf_counter() - t0) * 1000
    resp_preview = str(result.get("response", ""))[:80].replace("\n", " ")
    log.info(f"GENERATE  done  [{elapsed:.0f}ms]  response='{resp_preview}...'")
    return result


@app.post("/api/chat")
async def chat(request: Request):
    _check_auth(request)
    body = await request.json()
    model = _resolve_model(body)
    _check_allowed(model)
    messages = body.get("messages", [])
    last_msg = str(messages[-1].get("content", "") if messages else "")[:80].replace("\n", " ")
    log.info(f"CHAT      model='{model}'  messages={len(messages)}  last='{last_msg}...'")
    await _ensure_model(model)
    body["model"] = model
    t0 = time.perf_counter()
    result = await _ollama_post("/api/chat", body)
    elapsed = (time.perf_counter() - t0) * 1000
    reply_preview = str(result.get("message", {}).get("content", ""))[:80].replace("\n", " ")
    log.info(f"CHAT      done  [{elapsed:.0f}ms]  reply='{reply_preview}...'")
    return result


@app.post("/v1/chat/completions")
async def openai_compatible(request: Request):
    _check_auth(request)
    body = await request.json()
    model = _resolve_model(body)
    _check_allowed(model)
    messages = body.get("messages", [])
    last_msg = str(messages[-1].get("content", "") if messages else "")[:80].replace("\n", " ")
    log.info(f"OAI-CHAT  model='{model}'  messages={len(messages)}  last='{last_msg}...'")
    await _ensure_model(model)
    body["model"] = model
    t0 = time.perf_counter()
    result = await _ollama_post("/v1/chat/completions", body)
    elapsed = (time.perf_counter() - t0) * 1000
    try:
        reply_preview = result["choices"][0]["message"]["content"][:80].replace("\n", " ")
    except (KeyError, IndexError):
        reply_preview = "(no content)"
    log.info(f"OAI-CHAT  done  [{elapsed:.0f}ms]  reply='{reply_preview}...'")
    return result