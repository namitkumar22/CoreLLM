"""
CoreLLM – Modal deployment (cost-optimised, no auth)

Architecture
------------
  Single service: 4× NVIDIA A10G (96 GB VRAM total)

  All models on one endpoint:
    "qwen3.6:35b"           – coding, text, tools, thinking,        context=256k
    "qwen3-vl:32b"          – vision, text, tools, thinking,        context=256k
    "nemotron3:33b"         – audio, text, tools, vision, thinking
    "lfm2.5-thinking:1.2b"  – ultra fast, tools, thinking,         context=32k
    "qwen3-embedding:8b"    – embedding

Resource strategy
-----------------
  ┌─ @enter (cold start) ─────────────────────────────────────────┐
  │  • Start Ollama server                                         │
  │  • Verify all models are on disk (pulled once via seeder job)  │
  │  • Nothing loaded into VRAM yet → snapshot captures this state │
  └────────────────────────────────────────────────────────────────┘
  ┌─ On request for model X ──────────────────────────────────────┐
  │  • If X already in VRAM → serve immediately (0 overhead)      │
  │  • If different model Y in VRAM → unload Y, load X, serve     │
  │  • Only ONE model lives in VRAM at a time                      │
  │    → 4×A10G (96 GB) can hold any model comfortably            │
  └────────────────────────────────────────────────────────────────┘

Cost controls
-------------
  min_containers=0       → $0 when idle (containers exist only during calls)
  scaledown_window=60    → check idle every 60 s for fine-grained billing
  container_idle_timeout → no explicit timeout; Modal default keeps container 5 min warm
  enable_memory_snapshot → Ollama server state is snapshotted;
                           cold start = resume from snapshot (~5s) + on-demand model load

All endpoints are open — no API key required.
"""

import os
import time
import subprocess
import asyncio
import logging

import modal
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# ── Modal primitives ───────────────────────────────────────────────────────────

# Persistent volume: model blobs live here, survive forever across cold starts.
# Populated ONCE by running:  modal run modal_app.py::pull_all_models
model_volume = modal.Volume.from_name("corellm-models", create_if_missing=True)

VOLUME_MOUNT = "/models"
OLLAMA_URL   = "http://127.0.0.1:11434"

# All 5 models
ALL_MODELS = [
    "qwen3.6:35b",           # coding, text, tools, thinking, 256k ctx
    "qwen3-vl:32b",          # vision, text, tools, thinking, 256k ctx
    "nemotron3:33b",         # audio, text, tools, vision, thinking
    "lfm2.5-thinking:1.2b",  # ultra fast, tools, thinking, 32k ctx
    "qwen3-embedding:8b",    # embedding
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("fastapi", "uvicorn[standard]", "httpx")
)

app = modal.App("corellm")

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  │  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("corellm")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _start_ollama() -> subprocess.Popen:
    """Start Ollama as a background process and block until it's ready."""
    import httpx as _httpx

    env = {
        **os.environ,
        "OLLAMA_MODELS": VOLUME_MOUNT,
        "OLLAMA_HOME":   "/tmp/ollama_home",
        # Let Ollama use all 4 GPUs for the active model
        "OLLAMA_NUM_PARALLEL": "1",   # 1 request at a time — maximises VRAM per model
        "OLLAMA_MAX_LOADED_MODELS": "1",  # hard cap: only 1 model in VRAM at a time
    }
    os.makedirs("/tmp/ollama_home", exist_ok=True)

    log.info("Starting Ollama server...")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for attempt in range(30):
        try:
            if _httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2).status_code == 200:
                log.info(f"Ollama ready ({attempt + 1}s) — 0 models in VRAM")
                return proc
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Ollama did not start within 30 s")


def _verify_models_on_disk():
    """
    Verify all models are present in the volume.
    Logs a warning for any missing ones (run pull_all_models to fix).
    Does NOT load anything into VRAM.
    """
    env    = {**os.environ, "OLLAMA_MODELS": VOLUME_MOUNT}
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, env=env)
    on_disk = result.stdout

    for model in ALL_MODELS:
        tag = model.split(":")[0]
        if tag in on_disk or model in on_disk:
            log.info(f"  ✓ '{model}' on disk")
        else:
            log.warning(f"  ✗ '{model}' NOT on disk — run: modal run modal_app.py::pull_all_models")


# ── Gateway factory ───────────────────────────────────────────────────────────

def _build_gateway() -> FastAPI:
    """
    FastAPI gateway that proxies Ollama with on-demand VRAM management.
    Models are loaded into VRAM only when first requested.
    Only one model lives in VRAM at a time.
    """
    import httpx as _httpx

    web = FastAPI(title="CoreLLM", version="2.0.0")

    # Track which model is currently in VRAM
    _active: dict = {"model": None}
    # Serialize model switches so concurrent requests don't race
    _lock = asyncio.Lock()

    # ── Middleware ─────────────────────────────────────────────────────────────

    @web.middleware("http")
    async def _log_requests(request: Request, call_next):
        t = time.perf_counter()
        log.info(f"→ {request.method} {request.url.path}")
        try:
            resp = await call_next(request)
        except Exception as exc:
            log.error(f"✗ {request.url.path}: {exc}")
            raise
        ms = (time.perf_counter() - t) * 1000
        level = logging.WARNING if resp.status_code >= 400 else logging.INFO
        log.log(level, f"← {request.url.path}  [{resp.status_code}]  {ms:.0f}ms")
        return resp

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _check_allowed(model: str):
        if model not in ALL_MODELS:
            raise HTTPException(
                status_code=403,
                detail=f"Model '{model}' not available. Allowed: {ALL_MODELS}",
            )

    def _resolve_model(body: dict) -> str:
        m = body.get("model", "").strip() or _active["model"]
        if not m:
            raise HTTPException(
                status_code=400,
                detail="No model specified. Pass 'model' in the request body.",
            )
        return m

    async def _ollama_post(path: str, body: dict) -> dict:
        async with _httpx.AsyncClient(timeout=600) as c:
            r = await c.post(f"{OLLAMA_URL}{path}", json=body)
            r.raise_for_status()
            return r.json()

    async def _unload_from_vram(model: str):
        """Release model from VRAM. It stays on disk — no re-download needed."""
        log.info(f"VRAM  unloading '{model}'...")
        try:
            async with _httpx.AsyncClient(timeout=30) as c:
                await c.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": 0},
                )
            log.info(f"VRAM  '{model}' released")
        except Exception as e:
            log.warning(f"VRAM  unload failed (non-fatal): {e}")

    async def _load_into_vram(model: str):
        """
        Load model from disk into VRAM on-demand.
        keep_alive=-1 → stays resident until we explicitly unload it.
        Spans all 4 A10G GPUs automatically via Ollama's CUDA backend.
        """
        log.info(f"VRAM  loading '{model}' on demand (from disk)...")
        t = time.perf_counter()
        async with _httpx.AsyncClient(timeout=600) as c:
            await c.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": -1},
            )
        log.info(f"VRAM  '{model}' ready  [{(time.perf_counter()-t)*1000:.0f}ms]")

    async def _ensure_in_vram(model: str):
        """
        Guarantee 'model' is in VRAM before serving a request.
        - Already active → instant return (no overhead)
        - Different model active → unload it, load requested model
        Uses a lock so concurrent requests serialize here, not race.
        """
        # Fast path — no lock needed if already active
        if _active["model"] == model:
            return

        async with _lock:
            # Re-check inside lock: another coroutine may have switched already
            if _active["model"] == model:
                return

            prev = _active["model"]
            if prev:
                log.info(f"SWITCH  '{prev}' → '{model}'")
                await _unload_from_vram(prev)
            else:
                log.info(f"LOAD  first request → loading '{model}' into VRAM")

            await _load_into_vram(model)
            _active["model"] = model

    # ── Exception handler ──────────────────────────────────────────────────────

    @web.exception_handler(Exception)
    async def _handle_error(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        log.exception(f"Unhandled: {exc}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ── Health ─────────────────────────────────────────────────────────────────

    @web.get("/")
    def health():
        return {
            "status": "ok",
            "active_model": _active["model"],   # None = nothing in VRAM yet
            "available_models": ALL_MODELS,
        }

    @web.get("/ping")
    def ping():
        return {"status": "alive"}

    @web.get("/api/models")
    async def list_models():
        async with _httpx.AsyncClient(timeout=30) as c:
            data = (await c.get(f"{OLLAMA_URL}/api/tags")).json()
        on_disk = [m["name"] for m in data.get("models", [])]
        return {
            "models": on_disk,
            "active_in_vram": _active["model"],
        }

    # ── Model lifecycle ────────────────────────────────────────────────────────

    @web.post("/api/load")
    async def load_model(request: Request):
        """Explicitly pre-warm a model into VRAM before sending inference requests."""
        body  = await request.json()
        model = body.get("model", "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="'model' field required")
        _check_allowed(model)
        await _ensure_in_vram(model)
        return {"status": "loaded", "model": model}

    @web.post("/api/unload")
    async def unload_model(request: Request):
        """Release a model from VRAM. It stays on disk for fast next load."""
        body  = await request.json()
        model = body.get("model", "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="'model' field required")
        await _unload_from_vram(model)
        if _active["model"] == model:
            _active["model"] = None
        return {"status": "unloaded", "model": model}

    @web.post("/api/switch")
    async def switch_model(request: Request):
        """Explicitly switch the active VRAM model (previous is unloaded)."""
        body      = await request.json()
        new_model = body.get("model", "").strip()
        if not new_model:
            raise HTTPException(status_code=400, detail="'model' field required")
        _check_allowed(new_model)
        prev = _active["model"]
        await _ensure_in_vram(new_model)
        return {"status": "switched", "previous_model": prev, "active_model": new_model}

    # ── Inference ──────────────────────────────────────────────────────────────

    @web.post("/api/generate")
    async def generate(request: Request):
        body  = await request.json()
        model = _resolve_model(body)
        _check_allowed(model)
        await _ensure_in_vram(model)   # loads on demand if not already warm
        body["model"] = model
        return await _ollama_post("/api/generate", body)

    @web.post("/api/chat")
    async def chat(request: Request):
        body  = await request.json()
        model = _resolve_model(body)
        _check_allowed(model)
        await _ensure_in_vram(model)
        body["model"] = model
        return await _ollama_post("/api/chat", body)

    @web.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        body  = await request.json()
        model = _resolve_model(body)
        _check_allowed(model)
        await _ensure_in_vram(model)
        body["model"] = model
        return await _ollama_post("/v1/chat/completions", body)

    return web


# ══════════════════════════════════════════════════════════════════════════════
# ── CoreLLM service — 4× A10G ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.cls(
    image=image,
    gpu=modal.gpu.A10G(count=4),   # 4× 24 GB = 96 GB VRAM — fits any model
    volumes={VOLUME_MOUNT: model_volume},

    # ── Cost: pay only while handling requests ─────────────────────────────────
    min_containers=0,              # no idle containers — $0 when nobody is calling
    scaledown_window=60,           # check for idle every 60 s (fine-grained billing)

    # ── Speed: GPU memory snapshot ─────────────────────────────────────────────
    # Snapshot captures: Ollama server running + 0 models in VRAM.
    # Cold start resumes from snapshot (~5 s) then loads the requested model on demand.
    enable_memory_snapshot=True,

    timeout=900,                   # 15 min max per request (for large model loads + long gen)
)
class CoreLLM:

    @modal.enter(snap=True)
    def startup(self):
        """
        Cold-start initialisation — included in GPU memory snapshot.
        1. Start Ollama server (uses all 4 GPUs via CUDA)
        2. Verify models are on disk (from persistent volume)
        3. Nothing loaded into VRAM — models load on first user request
        """
        self._proc = _start_ollama()
        log.info("Checking model disk cache...")
        _verify_models_on_disk()
        log.info("Ready — VRAM is empty, models will load on first request")
        self._gateway = _build_gateway()

    @modal.exit()
    def shutdown(self):
        """Gracefully stop Ollama when the container exits."""
        if hasattr(self, "_proc"):
            self._proc.terminate()

    @modal.asgi_app(label="corellm")
    def web(self):
        return self._gateway


# ══════════════════════════════════════════════════════════════════════════════
# ── One-time model seeder ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu=modal.gpu.A10G(count=1),   # Only 1 GPU needed for downloading
    volumes={VOLUME_MOUNT: model_volume},
    timeout=7200,                  # 2 hr max — large models take time to download
)
def pull_all_models():
    """
    Seeds ALL models into the persistent volume.
    Run ONCE before deploying (or after adding new models).

        modal run modal_app.py::pull_all_models

    After this, containers NEVER re-download — they read from disk.
    """
    proc = _start_ollama()
    env  = {**os.environ, "OLLAMA_MODELS": VOLUME_MOUNT}

    log.info(f"Seeding {len(ALL_MODELS)} models into persistent volume...")
    for model in ALL_MODELS:
        # Check if already cached to avoid re-downloading
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, env=env)
        tag = model.split(":")[0]
        if tag in result.stdout or model in result.stdout:
            log.info(f"  ✓ '{model}' already cached — skipping")
            continue
        log.info(f"  → downloading '{model}'...")
        subprocess.run(["ollama", "pull", model], check=True, env=env)
        log.info(f"  ✓ '{model}' cached")

    model_volume.commit()  # flush writes to persistent storage
    proc.terminate()
    log.info("All models seeded. Volume is ready for deployment.")
    return {"status": "done", "models": ALL_MODELS}
