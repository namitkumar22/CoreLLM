"""
CoreLLM – Modal deployment (cost-optimised, no auth)

Architecture
------------
  Single service: 4× NVIDIA A10G (96 GB VRAM total, ~$1.60/hr)

  All models on one endpoint:
    "qwen3.6:35b"           – coding, text, tools, thinking,        context=256k
    "qwen3-vl:32b"          – vision, text, tools, thinking,        context=256k
    "nemotron3:33b"         – audio, text, tools, vision, thinking
    "lfm2.5-thinking:1.2b"  – ultra fast, tools, thinking,         context=32k
    "qwen3-embedding:8b"    – embedding

Concurrent-user strategy (two layers)
--------------------------------------
  Layer 1 — Within one container (96 GB VRAM):
    OLLAMA_MAX_LOADED_MODELS=3 → Ollama keeps up to 3 models warm in VRAM
    simultaneously using its built-in LRU cache.  When a 4th model is needed,
    Ollama automatically evicts the least-recently-used one.  No manual locking.
    Example: User A on qwen3.6:35b (~22 GB) + User B on qwen3-vl:32b (~20 GB)
    + User C on lfm2.5:1.2b (~1 GB) = ~43 GB  →  all fit in 96 GB. ✅

  Layer 2 — Across containers (Modal autoscale):
    If concurrent demand exceeds what one container can handle, Modal spins up
    additional containers automatically (each gets its own 4×A10G).
    You pay per-second only for containers that are actually running.

  Adding new models: just add the name to ALL_MODELS and run pull_all_models.
  No other code change needed — Ollama + Modal handle the rest.

Cost controls
-------------
  min_containers=0       → $0 when idle — no containers exist between calls
  scaledown_window=60    → idle check every 60 s for fine-grained billing
  enable_memory_snapshot → Ollama server state checkpointed; cold start ~5 s
                           (models still load from disk on first request per container)
  OLLAMA_MAX_LOADED_MODELS=3 → caps VRAM usage; LRU eviction handles overflow
  OLLAMA_NUM_PARALLEL=4  → up to 4 concurrent requests per container

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

model_volume = modal.Volume.from_name("corellm-models", create_if_missing=True)

VOLUME_MOUNT = "/models"
OLLAMA_URL   = "http://127.0.0.1:11434"

# ── Model registry ─────────────────────────────────────────────────────────────
# To add a new model: append it here, then run:  modal run modal_app.py::pull_all_models
ALL_MODELS: list[str] = [
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
    """Start Ollama and wait until it's ready. Uses all 4 GPUs via CUDA."""
    import httpx as _httpx

    env = {
        **os.environ,
        "OLLAMA_MODELS":            VOLUME_MOUNT,
        "OLLAMA_HOME":              "/tmp/ollama_home",
        # ── Concurrency settings ───────────────────────────────────────────────
        # Keep up to 3 models warm in VRAM simultaneously (LRU eviction for 4th+)
        # 3 × ~22 GB ≈ 66 GB  →  comfortably fits in 96 GB
        "OLLAMA_MAX_LOADED_MODELS": "3",
        # Allow 4 parallel inference requests per container
        "OLLAMA_NUM_PARALLEL":      "4",
        # How long a loaded model stays in VRAM with no requests (10 min)
        "OLLAMA_KEEP_ALIVE":        "10m",
    }
    os.makedirs("/tmp/ollama_home", exist_ok=True)

    log.info("Starting Ollama (4× A10G, max 3 models in VRAM)...")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for attempt in range(30):
        try:
            if _httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2).status_code == 200:
                log.info(f"Ollama ready ({attempt + 1}s)")
                return proc
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Ollama did not start within 30 s")


def _verify_models_on_disk():
    """Log which models are on disk vs missing. No VRAM loading."""
    env    = {**os.environ, "OLLAMA_MODELS": VOLUME_MOUNT}
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, env=env)
    on_disk = result.stdout
    for model in ALL_MODELS:
        tag = model.split(":")[0]
        if tag in on_disk or model in on_disk:
            log.info(f"  ✓ '{model}' on disk")
        else:
            log.warning(f"  ✗ '{model}' missing — run: modal run modal_app.py::pull_all_models")


# ── Gateway factory ───────────────────────────────────────────────────────────

def _build_gateway() -> FastAPI:
    """
    FastAPI app that proxies Ollama.

    Model VRAM management is fully delegated to Ollama:
      - Ollama loads a model from disk on first request (automatic)
      - Keeps up to OLLAMA_MAX_LOADED_MODELS warm simultaneously (LRU cache)
      - Evicts least-recently-used model when VRAM cap is hit (automatic)
      - No manual locking or switching needed — concurrent requests just work
    """
    import httpx as _httpx

    web = FastAPI(title="CoreLLM", version="2.0.0")

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
                detail=f"Model '{model}' is not available. Allowed: {ALL_MODELS}",
            )

    def _resolve_model(body: dict) -> str:
        m = body.get("model", "").strip()
        if not m:
            raise HTTPException(
                status_code=400,
                detail="No 'model' specified in request body.",
            )
        return m

    async def _proxy(path: str, body: dict) -> dict:
        """Forward request to Ollama. Ollama handles VRAM loading automatically."""
        async with _httpx.AsyncClient(timeout=600) as c:
            r = await c.post(f"{OLLAMA_URL}{path}", json=body)
            r.raise_for_status()
            return r.json()

    # ── Exception handler ──────────────────────────────────────────────────────

    @web.exception_handler(Exception)
    async def _handle_error(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        log.exception(f"Unhandled: {exc}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ── Health & info ──────────────────────────────────────────────────────────

    @web.get("/")
    async def health():
        """Returns server status and which models are currently loaded in VRAM."""
        try:
            async with _httpx.AsyncClient(timeout=5) as c:
                ps = (await c.get(f"{OLLAMA_URL}/api/ps")).json()
            loaded = [m["name"] for m in ps.get("models", [])]
        except Exception:
            loaded = []
        return {
            "status":           "ok",
            "models_in_vram":   loaded,        # what's hot right now
            "available_models": ALL_MODELS,    # everything on disk
            "vram_slots":       "3 (LRU)",     # max simultaneous models
        }

    @web.get("/ping")
    def ping():
        return {"status": "alive"}

    @web.get("/api/models")
    async def list_models():
        """List all models on disk + which are currently in VRAM."""
        async with _httpx.AsyncClient(timeout=30) as c:
            tags = (await c.get(f"{OLLAMA_URL}/api/tags")).json()
            try:
                ps = (await c.get(f"{OLLAMA_URL}/api/ps")).json()
                in_vram = [m["name"] for m in ps.get("models", [])]
            except Exception:
                in_vram = []
        on_disk = [m["name"] for m in tags.get("models", [])]
        return {
            "models":       on_disk,
            "models_in_vram": in_vram,
        }

    # ── Manual VRAM control (optional — Ollama manages this automatically) ─────

    @web.post("/api/load")
    async def load_model(request: Request):
        """
        Pre-warm a model into VRAM before sending inference requests.
        Optional — Ollama loads models automatically on first inference call.
        Useful to avoid first-request latency for specific models.
        """
        body  = await request.json()
        model = body.get("model", "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="'model' field required")
        _check_allowed(model)
        # Trigger a no-op generate to force load into VRAM
        async with _httpx.AsyncClient(timeout=600) as c:
            await c.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": model, "prompt": "", "keep_alive": "10m"})
        log.info(f"LOAD  '{model}' pre-warmed into VRAM")
        return {"status": "loaded", "model": model}

    @web.post("/api/unload")
    async def unload_model(request: Request):
        """Explicitly release a model from VRAM (it stays on disk)."""
        body  = await request.json()
        model = body.get("model", "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="'model' field required")
        async with _httpx.AsyncClient(timeout=30) as c:
            await c.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": model, "prompt": "", "keep_alive": 0})
        log.info(f"UNLOAD  '{model}' released from VRAM")
        return {"status": "unloaded", "model": model}

    # ── Inference ──────────────────────────────────────────────────────────────
    # No manual model switching needed — Ollama loads/evicts automatically per request.

    @web.post("/api/generate")
    async def generate(request: Request):
        body  = await request.json()
        model = _resolve_model(body)
        _check_allowed(model)
        return await _proxy("/api/generate", body)

    @web.post("/api/chat")
    async def chat(request: Request):
        body  = await request.json()
        model = _resolve_model(body)
        _check_allowed(model)
        return await _proxy("/api/chat", body)

    @web.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        body  = await request.json()
        model = _resolve_model(body)
        _check_allowed(model)
        return await _proxy("/v1/chat/completions", body)

    return web


# ══════════════════════════════════════════════════════════════════════════════
# ── CoreLLM service — 4× A10G ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.cls(
    image=image,
    gpu=modal.gpu.A10G(count=4),   # 4× 24 GB = 96 GB VRAM total
    volumes={VOLUME_MOUNT: model_volume},

    # ── Cost controls ──────────────────────────────────────────────────────────
    min_containers=0,              # $0 when idle — no containers exist between calls
    scaledown_window=60,           # fine-grained idle billing (check every 60 s)

    # ── Cold-start optimisation ────────────────────────────────────────────────
    # Snapshot = Ollama running, 0 models in VRAM.
    # Resume from snapshot (~5 s) + model loads from disk on first request.
    enable_memory_snapshot=True,

    # ── Autoscaling ────────────────────────────────────────────────────────────
    # Modal automatically scales up containers when concurrent demand increases.
    # Each new container gets its own 4×A10G and its own Ollama instance.
    # max_containers is unset = unlimited scaling (bounded by your credit).

    timeout=900,   # 15 min max per request
)
class CoreLLM:

    @modal.enter(snap=True)
    def startup(self):
        """
        Cold-start init — captured in GPU memory snapshot.
          1. Start Ollama (all 4 GPUs available via CUDA)
          2. Verify models are on disk
          3. 0 models in VRAM — they load on first user request per model
        """
        self._proc = _start_ollama()
        log.info("Verifying model disk cache...")
        _verify_models_on_disk()
        log.info("Startup complete — VRAM empty, models load on demand")
        self._gateway = _build_gateway()

    @modal.exit()
    def shutdown(self):
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
    gpu=modal.gpu.A10G(count=1),   # 1 GPU sufficient for downloading
    volumes={VOLUME_MOUNT: model_volume},
    timeout=7200,                  # 2 hr max (large models take time)
)
def pull_all_models():
    """
    Download all models in ALL_MODELS into the persistent volume.
    Run ONCE before first deploy, and again whenever you add new models.

        modal run modal_app.py::pull_all_models

    After seeding, containers read from disk on startup — no re-downloading.
    To add a new model: append to ALL_MODELS above, then re-run this function.
    """
    proc = _start_ollama()
    env  = {**os.environ, "OLLAMA_MODELS": VOLUME_MOUNT}

    log.info(f"Seeding {len(ALL_MODELS)} models into persistent volume...")
    for model in ALL_MODELS:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, env=env)
        tag = model.split(":")[0]
        if tag in result.stdout or model in result.stdout:
            log.info(f"  ✓ '{model}' already cached — skipping")
            continue
        log.info(f"  → downloading '{model}'...")
        subprocess.run(["ollama", "pull", model], check=True, env=env)
        log.info(f"  ✓ '{model}' done")

    model_volume.commit()
    proc.terminate()
    log.info("All models cached. Ready to deploy.")
    return {"status": "done", "models": ALL_MODELS}
