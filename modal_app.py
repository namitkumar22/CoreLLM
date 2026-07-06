"""
CoreLLM – Modal deployment (cost-optimised, no auth)

Architecture
------------
  CoreLLMHeavy  → NVIDIA H200  (141 GB VRAM) — nemotron-3-super:120b, qwen3-vl:32b
  CoreLLMLight  → NVIDIA A10G  (24 GB VRAM)  — lfm2.5-thinking:1.2b, qwen3-embedding:8b

Both tiers share one persistent Modal Volume ("corellm-models") so models are
downloaded exactly ONCE and reused across all cold starts.

Cost controls
-------------
  min_containers=0       → $0 when idle — containers only exist during active calls
  container_idle_timeout → 5 min warm window after last request (avoids repeated cold starts)
  enable_memory_snapshot → GPU state is checkpointed; cold start drops ~60 s → ~5 s
  A10G for light tier    → 3× faster than T4 at comparable cost for small models

All endpoints are open (no API key required).
"""

import os
import time
import subprocess
import asyncio
import logging
from typing import Optional

import modal
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# ── Modal primitives ───────────────────────────────────────────────────────────

# Persistent volume — model blobs live here, survive container restarts forever
model_volume = modal.Volume.from_name("corellm-models", create_if_missing=True)

VOLUME_MOUNT = "/models"  # where the volume is mounted inside every container
OLLAMA_URL   = "http://127.0.0.1:11434"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
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
                log.info(f"Ollama ready ({attempt + 1}s)")
                return proc
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Ollama did not start within 30 s")


def _pull_if_missing(model: str):
    """Pull a model into the persistent volume only if it isn't cached yet."""
    env = {**os.environ, "OLLAMA_MODELS": VOLUME_MOUNT}
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, env=env)
    tag = model.split(":")[0]
    if tag in result.stdout or model in result.stdout:
        log.info(f"'{model}' already cached — skipping pull")
        return
    log.info(f"Pulling '{model}' into volume (first time only)...")
    subprocess.run(["ollama", "pull", model], check=True, env=env)
    log.info(f"'{model}' cached ✓")


def _build_gateway(allowed_models: list[str]) -> FastAPI:
    """
    Build and return a FastAPI app that proxies Ollama with auto-model-switching.
    No authentication — fully open.
    """
    import httpx as _httpx

    web = FastAPI(title="CoreLLM", version="2.0.0")
    _active: dict = {"model": None}
    _lock = asyncio.Lock()

    # ── Middleware ─────────────────────────────────────────────────────────────

    @web.middleware("http")
    async def _log(request: Request, call_next):
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

    # ── Validation ─────────────────────────────────────────────────────────────

    def _check_allowed(model: str):
        if allowed_models and model not in allowed_models:
            raise HTTPException(
                status_code=403,
                detail=f"Model '{model}' not available. Allowed: {allowed_models}",
            )

    def _resolve_model(body: dict) -> str:
        m = body.get("model", "").strip() or _active["model"]
        if not m:
            raise HTTPException(
                status_code=400,
                detail="No model specified. Pass 'model' in the request body.",
            )
        return m

    # ── Ollama operations ──────────────────────────────────────────────────────

    async def _ollama_post(path: str, body: dict) -> dict:
        async with _httpx.AsyncClient(timeout=600) as c:
            r = await c.post(f"{OLLAMA_URL}{path}", json=body)
            r.raise_for_status()
            return r.json()

    async def _unload(model: str):
        log.info(f"UNLOAD '{model}'...")
        try:
            async with _httpx.AsyncClient(timeout=30) as c:
                await c.post(f"{OLLAMA_URL}/api/generate",
                             json={"model": model, "prompt": "", "keep_alive": 0})
        except Exception as e:
            log.warning(f"UNLOAD failed (non-fatal): {e}")

    async def _load(model: str):
        log.info(f"LOAD '{model}' into VRAM (keep_alive=-1)...")
        t = time.perf_counter()
        async with _httpx.AsyncClient(timeout=300) as c:
            await c.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": model, "prompt": "", "keep_alive": -1})
        log.info(f"LOAD '{model}' ready  [{(time.perf_counter()-t)*1000:.0f}ms]")

    async def _ensure(model: str):
        if _active["model"] == model:
            return
        async with _lock:
            if _active["model"] == model:
                return
            prev = _active["model"]
            log.info(f"SWITCH '{prev}' → '{model}'")
            if prev:
                await _unload(prev)
            await _load(model)
            _active["model"] = model

    # ── Exception handler ──────────────────────────────────────────────────────

    @web.exception_handler(Exception)
    async def _err(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        log.exception(f"Unhandled: {exc}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ── Health ─────────────────────────────────────────────────────────────────

    @web.get("/")
    def health():
        return {"status": "ok", "active_model": _active["model"], "allowed_models": allowed_models}

    @web.get("/ping")
    def ping():
        return {"status": "alive"}

    @web.get("/api/models")
    async def list_models():
        async with _httpx.AsyncClient(timeout=30) as c:
            data = (await c.get(f"{OLLAMA_URL}/api/tags")).json()
        all_local = [m["name"] for m in data.get("models", [])]
        visible = [m for m in all_local if not allowed_models or m in allowed_models]
        return {"models": visible, "active": _active["model"]}

    # ── Model lifecycle ────────────────────────────────────────────────────────

    @web.post("/api/load")
    async def load_model(request: Request):
        body  = await request.json()
        model = body.get("model", "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="'model' field required")
        _check_allowed(model)
        await _ensure(model)
        return {"status": "loaded", "model": model}

    @web.post("/api/unload")
    async def unload_model(request: Request):
        body  = await request.json()
        model = body.get("model", "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="'model' field required")
        await _unload(model)
        if _active["model"] == model:
            _active["model"] = None
        return {"status": "unloaded", "model": model}

    @web.post("/api/switch")
    async def switch_model(request: Request):
        body      = await request.json()
        new_model = body.get("model", "").strip()
        if not new_model:
            raise HTTPException(status_code=400, detail="'model' field required")
        _check_allowed(new_model)
        prev = _active["model"]
        await _ensure(new_model)
        return {"status": "switched", "previous_model": prev, "active_model": new_model}

    # ── Inference ──────────────────────────────────────────────────────────────

    @web.post("/api/generate")
    async def generate(request: Request):
        body  = await request.json()
        model = _resolve_model(body)
        _check_allowed(model)
        await _ensure(model)
        body["model"] = model
        return await _ollama_post("/api/generate", body)

    @web.post("/api/chat")
    async def chat(request: Request):
        body  = await request.json()
        model = _resolve_model(body)
        _check_allowed(model)
        await _ensure(model)
        body["model"] = model
        return await _ollama_post("/api/chat", body)

    @web.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        body  = await request.json()
        model = _resolve_model(body)
        _check_allowed(model)
        await _ensure(model)
        body["model"] = model
        return await _ollama_post("/v1/chat/completions", body)

    return web


# ══════════════════════════════════════════════════════════════════════════════
# ── Heavy tier — H200 ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

HEAVY_MODELS = [
    "nemotron-3-super:120b",  # 120B — primary model (text, tools, thinking, 256k ctx)
    "qwen3-vl:32b",           # 32B  — vision, text, tools, thinking, 256k ctx
]

@app.cls(
    image=image,
    gpu="H200",
    volumes={VOLUME_MOUNT: model_volume},
    # ── Cost: pay only when active ─────────────────────────────────────────────
    min_containers=0,
    scaledown_window=60,           # check idle every 60 s for fine-grained billing
    # ── Speed: GPU memory snapshot ─────────────────────────────────────────────
    # Checkpoint the VRAM state after startup → subsequent cold starts resume in ~5 s
    enable_memory_snapshot=True,
    timeout=900,                   # 15 min max per request (long generations)
)
class CoreLLMHeavy:

    @modal.enter(snap=True)
    def startup(self):
        """Runs once on cold start. snap=True → included in GPU memory snapshot."""
        import httpx as _httpx
        self._proc = _start_ollama()

        for m in HEAVY_MODELS:
            _pull_if_missing(m)

        # Pre-warm primary model into VRAM so the snapshot captures it hot
        log.info("Pre-warming nemotron-3-super:120b for GPU snapshot...")
        try:
            _httpx.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": "nemotron-3-super:120b", "prompt": "", "keep_alive": -1},
                timeout=300,
            )
            log.info("nemotron-3-super:120b hot in VRAM ✓")
        except Exception as e:
            log.warning(f"Pre-warm failed (non-fatal): {e}")

        self._gateway = _build_gateway(HEAVY_MODELS)

    @modal.exit()
    def shutdown(self):
        if hasattr(self, "_proc"):
            self._proc.terminate()

    @modal.asgi_app(label="corellm-heavy")
    def web(self):
        return self._gateway


# ══════════════════════════════════════════════════════════════════════════════
# ── Light tier — A10G ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

LIGHT_MODELS = [
    "lfm2.5-thinking:1.2b",  # ultra fast — tools, thinking, 32k ctx
    "qwen3-embedding:8b",    # embedding
]

@app.cls(
    image=image,
    gpu="A10G",   # 24 GB VRAM — ~3× faster than T4 for small models, still cost-effective
    volumes={VOLUME_MOUNT: model_volume},
    min_containers=0,
    scaledown_window=60,
    enable_memory_snapshot=True,
    timeout=300,
)
class CoreLLMLight:

    @modal.enter(snap=True)
    def startup(self):
        import httpx as _httpx
        self._proc = _start_ollama()

        for m in LIGHT_MODELS:
            _pull_if_missing(m)

        # Pre-warm the fast model into VRAM
        log.info("Pre-warming lfm2.5-thinking:1.2b for GPU snapshot...")
        try:
            _httpx.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": "lfm2.5-thinking:1.2b", "prompt": "", "keep_alive": -1},
                timeout=120,
            )
            log.info("lfm2.5-thinking:1.2b hot in VRAM ✓")
        except Exception as e:
            log.warning(f"Pre-warm failed (non-fatal): {e}")

        self._gateway = _build_gateway(LIGHT_MODELS)

    @modal.exit()
    def shutdown(self):
        if hasattr(self, "_proc"):
            self._proc.terminate()

    @modal.asgi_app(label="corellm-light")
    def web(self):
        return self._gateway


# ══════════════════════════════════════════════════════════════════════════════
# ── One-time model seeder — run manually to fill the volume ───────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A10G",           # A10G sufficient for downloading; saves H200 quota
    volumes={VOLUME_MOUNT: model_volume},
    timeout=3600,
)
def pull_all_models():
    """
    Seeds all models into the persistent volume (run ONCE on first deploy).

        modal run modal_app.py::pull_all_models

    After this, cold starts just read from disk — no re-downloading ever.
    """
    proc = _start_ollama()
    env  = {**os.environ, "OLLAMA_MODELS": VOLUME_MOUNT}

    all_models = HEAVY_MODELS + LIGHT_MODELS
    log.info(f"Seeding {len(all_models)} models into volume...")
    for m in all_models:
        log.info(f"  → pulling '{m}'...")
        subprocess.run(["ollama", "pull", m], check=True, env=env)
        log.info(f"  ✓ '{m}' cached")

    model_volume.commit()   # flush writes to persistent storage
    proc.terminate()
    log.info("All models seeded. Volume is ready.")
    return {"status": "done", "models": all_models}
