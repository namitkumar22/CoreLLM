"""
CoreLLM SDK  –  corellm-sdk
============================
A LangChain-native client for the CoreLLM inference gateway.
Drop-in replacement for ChatOpenAI / ChatGroq — no API key required.

Models
------
  Heavy tier (H200 — set base_url to the heavy endpoint):
    "nemotron-3-super:120b"  – text, tools, thinking,            context=256k  (primary)
    "qwen3-vl:32b"           – text, vision, tools, thinking,   context=256k

  Light tier (A10G — set base_url to the light endpoint):
    "lfm2.5-thinking:1.2b"  – ultra fast, tools, thinking, context=32k
    "qwen3-embedding:8b"    – embedding

Install
-------
    pip install corellm-sdk

Usage
-----
    from corellm_sdk import CoreLLMChat

    llm = CoreLLMChat(
        model="nemotron-3-super:120b",
        base_url="https://<your-workspace>--corellm-heavy-web.modal.run",
    )

    # LangChain
    from langchain_core.messages import HumanMessage
    reply = llm.invoke([HumanMessage(content="Hello!")])
    print(reply.content)

    # Raw chat
    print(llm.raw_chat([{"role": "user", "content": "Hello!"}]))

    # Raw generate
    print(llm.raw_generate("Explain quantum physics in one sentence."))

    # Embedding (use light endpoint)
    embed_llm = CoreLLMChat(
        model="qwen3-embedding:8b",
        base_url="https://<your-workspace>--corellm-light-web.modal.run",
    )
    print(embed_llm.embed("Some text to embed"))

Environment variables
---------------------
    CORELLM_BASE_URL   – overrides the default base_url at runtime
"""

from __future__ import annotations

import os
import httpx
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import Field


# Default endpoints — override with base_url or CORELLM_BASE_URL env var
_DEFAULT_HEAVY_URL = "https://namitkumar22--corellm-heavy-web.modal.run"
_DEFAULT_LIGHT_URL = "https://namitkumar22--corellm-light-web.modal.run"

# Convenience alias users can import to avoid typos
HEAVY_ENDPOINT = _DEFAULT_HEAVY_URL
LIGHT_ENDPOINT = _DEFAULT_LIGHT_URL


class CoreLLMChat(ChatOpenAI):
    """
    LangChain-compatible chat model backed by the CoreLLM inference gateway.

    Subclasses ChatOpenAI — supports everything LangChain and LangGraph provide:
    agents, bind_tools, LCEL chains, streaming, RAG, memory, etc.

    Parameters
    ----------
    model : str
        Model name.  See module docstring for the full list.
    base_url : str, optional
        Gateway URL.  Defaults to the heavy endpoint (nemotron-super / gemma4:31b).
        Use LIGHT_ENDPOINT for lfm2.5-thinking:1.2b or qwen3-embedding:8b.
    preload : bool
        Pre-warm the model on the server at init time (default True).
        Set to False for faster instantiation if the model is already warm.
    timeout : int
        HTTP request timeout in seconds (default 300).
    """

    _corellm_base: str = ""
    _corellm_timeout: int = 300

    def __init__(self, **kwargs: Any):
        # ── Resolve base URL ──────────────────────────────────────────────────
        env_url  = os.environ.get("CORELLM_BASE_URL", "")
        base_url = kwargs.pop("base_url", env_url or _DEFAULT_HEAVY_URL).rstrip("/")
        timeout  = kwargs.pop("timeout",  300)
        preload  = kwargs.pop("preload",  True)

        # ChatOpenAI requires a base_url ending in /v1 and a (dummy) api_key
        kwargs["base_url"]   = f"{base_url}/v1"
        kwargs["api_key"]    = "not-needed"        # gateway has no auth

        super().__init__(**kwargs)

        # Store our extras after super().__init__ to avoid Pydantic conflicts
        object.__setattr__(self, "_corellm_base",    base_url)
        object.__setattr__(self, "_corellm_timeout", timeout)

        if preload:
            self._preload(self.model_name)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @property
    def _llm_type(self) -> str:
        return "corellm_sdk"

    def _post(self, path: str, body: dict) -> dict:
        with httpx.Client(timeout=self._corellm_timeout) as c:
            r = c.post(f"{self._corellm_base}{path}", json=body)
            r.raise_for_status()
            return r.json()

    def _get(self, path: str) -> dict:
        with httpx.Client(timeout=self._corellm_timeout) as c:
            r = c.get(f"{self._corellm_base}{path}")
            r.raise_for_status()
            return r.json()

    def _preload(self, model: str):
        print(f"[CoreLLM] Pre-warming '{model}'...")
        try:
            self._post("/api/load", {"model": model})
            print(f"[CoreLLM] ✓ '{model}' is ready")
        except Exception as e:
            print(f"[CoreLLM] Pre-warm failed (server may be cold-starting): {e}")

    # ── Server control ─────────────────────────────────────────────────────────

    def switch(self, new_model: str) -> "CoreLLMChat":
        """
        Switch the active model on the server and update this instance.
        The previous model is automatically unloaded from VRAM.

        Example
        -------
            llm = CoreLLMChat(model="nemotron-super", ...)
            llm.switch("gemma4:31b")
        """
        print(f"[CoreLLM] Switching '{self.model_name}' → '{new_model}'...")
        self._post("/api/switch", {"model": new_model})
        self.model_name = new_model
        print(f"[CoreLLM] ✓ Active model is now '{new_model}'")
        return self

    def unload(self) -> dict:
        """Release the current model from server VRAM (keeps it on disk)."""
        result = self._post("/api/unload", {"model": self.model_name})
        print(f"[CoreLLM] '{self.model_name}' unloaded from VRAM")
        return result

    def list_models(self) -> list[str]:
        """Return all models available on this server endpoint."""
        return self._get("/api/models").get("models", [])

    def status(self) -> dict:
        """Return server health, active model, and allowed models."""
        return self._get("/")

    # ── Inference ──────────────────────────────────────────────────────────────

    def raw_chat(self, messages: list[dict], **kwargs) -> str:
        """
        Multi-turn chat via the Ollama /api/chat endpoint.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style messages: [{"role": "user", "content": "..."}]
        **kwargs
            Extra Ollama parameters (temperature, num_predict, etc.)

        Returns
        -------
        str  — the assistant's reply text
        """
        body   = {"model": self.model_name, "messages": messages, "stream": False, **kwargs}
        result = self._post("/api/chat", body)
        return result.get("message", {}).get("content", "")

    def raw_generate(self, prompt: str, **kwargs) -> str:
        """
        Raw text completion via the Ollama /api/generate endpoint.

        Parameters
        ----------
        prompt : str
        **kwargs
            Extra Ollama parameters (temperature, num_predict, etc.)

        Returns
        -------
        str  — the generated text
        """
        body   = {"model": self.model_name, "prompt": prompt, "stream": False, **kwargs}
        result = self._post("/api/generate", body)
        return result.get("response", "")

    def openai_chat(self, messages: list[dict], **kwargs) -> str:
        """
        OpenAI-compatible chat via /v1/chat/completions.
        Use this when you need the raw OpenAI response format.

        Returns
        -------
        str  — the assistant's reply text
        """
        body   = {"model": self.model_name, "messages": messages, **kwargs}
        result = self._post("/v1/chat/completions", body)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return str(result)

    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for the given text.
        Only meaningful when model is "qwen3-embedding:8b".

        Returns
        -------
        list[float]  — embedding vector
        """
        result = self._post("/api/generate", {
            "model":  self.model_name,
            "prompt": text,
            "stream": False,
        })
        # Ollama returns embeddings under the "embedding" key when available
        return result.get("embedding", result.get("response", []))
