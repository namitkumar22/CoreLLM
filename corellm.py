"""
CoreLLM Python Client SDK
=========================
A clean Python interface to your CoreLLM Hugging Face Space.

All models must be pre-baked into the Space image at build time.
The client will auto-switch models on the server as needed —
you never need to call .switch() manually unless you want to
pre-warm a model before the first inference call.

Usage
-----
    from corellm import CoreLLM

    llm = CoreLLM(
        model="qwen2.5:1.5b",
        base_url="https://namitkumar22-core-llm-api.hf.space",
        api_key="your-secret-key",   # omit if API_KEY is not set on server
    )

    # Generate text
    print(llm.generate("Explain gravity in one sentence."))

    # Chat
    reply = llm.chat([{"role": "user", "content": "Hello!"}])
    print(reply)

    # Use a different model — previous is automatically unloaded on the server
    llm.model = "llama3.2:3b"
    print(llm.generate("Who are you?"))  # server auto-switches
"""

from __future__ import annotations

import os
import httpx
from typing import Optional


class CoreLLM:
    """
    Client for the CoreLLM API gateway.

    Parameters
    ----------
    model : str
        The default model to use. Must be in the server's ALLOWED_MODELS list.
    base_url : str, optional
        Base URL of your CoreLLM Space.
        Falls back to CORELLM_BASE_URL environment variable.
    api_key : str, optional
        Bearer token. Falls back to CORELLM_API_KEY environment variable.
    preload : bool
        If True (default), send a /api/load request so the model is warm
        before the first inference call. Set to False to skip the pre-warm.
    timeout : int
        Request timeout in seconds (default 300).
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        preload: bool = True,
        timeout: int = 300,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("CORELLM_BASE_URL", "")).rstrip("/")
        self._api_key = api_key or os.environ.get("CORELLM_API_KEY", "")
        self.timeout = timeout

        if not self.base_url:
            raise ValueError(
                "base_url is required. Pass it directly or set CORELLM_BASE_URL."
            )

        if preload:
            self._preload(model)

    # ── Internals ─────────────────────────────────────────────────────────────

    @property
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _post(self, path: str, body: dict) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}{path}", json=body, headers=self._headers)
            r.raise_for_status()
            return r.json()

    def _get(self, path: str) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(f"{self.base_url}{path}", headers=self._headers)
            r.raise_for_status()
            return r.json()

    def _preload(self, model: str):
        """Ask the server to pre-warm the model so the first request is fast."""
        print(f"[CoreLLM] Pre-warming '{model}' on server...")
        self._post("/api/load", {"model": model})
        print(f"[CoreLLM] ✓ '{model}' is ready.")

    # ── Model control ─────────────────────────────────────────────────────────

    def switch(self, new_model: str) -> "CoreLLM":
        """
        Explicitly switch the active model on the server.
        The previous model is unloaded from RAM; the new one is loaded.
        Returns self for chaining.

        Note: You can also just set llm.model = "new-model-name" and the
        server will auto-switch on the next inference call.
        """
        print(f"[CoreLLM] Switching '{self.model}' → '{new_model}'...")
        result = self._post("/api/switch", {"model": new_model})
        self.model = new_model
        print(f"[CoreLLM] ✓ Active model is now '{new_model}'.")
        return self

    def unload(self) -> dict:
        """Release the current model from server RAM (stays on disk)."""
        result = self._post("/api/unload", {"model": self.model})
        print(f"[CoreLLM] '{self.model}' unloaded from memory.")
        return result

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Raw text completion. Server auto-switches model if self.model changed.

        Parameters
        ----------
        prompt : str
            The input prompt.
        **kwargs
            Extra Ollama /api/generate params (e.g. options={"temperature": 0.7}).

        Returns
        -------
        str
            The generated text.
        """
        body = {"model": self.model, "prompt": prompt, "stream": False, **kwargs}
        result = self._post("/api/generate", body)
        return result.get("response", "")

    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Multi-turn chat. Server auto-switches model if self.model changed.

        Parameters
        ----------
        messages : list of dict
            e.g. [{"role": "user", "content": "Hello"}]
        **kwargs
            Extra Ollama /api/chat params.

        Returns
        -------
        str
            The assistant's reply.
        """
        body = {"model": self.model, "messages": messages, "stream": False, **kwargs}
        result = self._post("/api/chat", body)
        return result.get("message", {}).get("content", "")

    def openai_chat(self, messages: list[dict], **kwargs) -> str:
        """
        OpenAI-compatible /v1/chat/completions endpoint.
        Compatible with any tool expecting the OpenAI response format.
        """
        body = {"model": self.model, "messages": messages, **kwargs}
        result = self._post("/v1/chat/completions", body)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return str(result)

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_models(self) -> list[str]:
        """Return all models available on the server (on disk + allowed)."""
        return self._get("/api/models").get("models", [])

    def status(self) -> dict:
        """Return server health, active model, and allowed models."""
        return self._get("/")

    def __repr__(self) -> str:
        return f"CoreLLM(model={self.model!r}, base_url={self.base_url!r})"
