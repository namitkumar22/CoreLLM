"""
CoreLLM SDK Python Client
=========================
A LangChain-native chat model client for your CoreLLM Hugging Face Space.

Usage
-----
    from corellm_sdk import CoreLLMChat
    from langchain_core.messages import HumanMessage
    
    llm = CoreLLMChat(
        model="gemma4:e4b"
    )
    
    # LangChain usage (Now supports EVERYTHING ChatOpenAI supports: agents, bind_tools, RAG, etc)
    response = llm.invoke([HumanMessage(content="Hello!")])
    print(response.content)
    
    # OpenAI format usage
    response = llm.openai_chat([{"role": "user", "content": "Hello!"}])
    print(response)
"""

from __future__ import annotations

import os
import httpx
from typing import Optional, Any, List, Mapping, Dict

# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI
from pydantic import Field


class CoreLLMChat(ChatOpenAI):
    """
    LangChain-compatible chat model backed by your CoreLLM HF Space.

    Subclasses ChatOpenAI — drop-in replacement that natively supports
    EVERYTHING in LangChain and LangGraph (agents, bind_tools, streaming).
    Also includes raw OpenAI compatibility methods.
    
    Parameters
    ----------
    model : str
        The model to use (must be in server's ALLOWED_MODELS).
    base_url : str, optional
        Your CoreLLM Space URL. Defaults to the public HF Space endpoint.
    preload : bool
        Pre-warm the model on init (default True).
    timeout : int
        Request timeout seconds (default 300).
    """

    corellm_base_url: str = Field(default="https://namitkumar22-corellm.hf.space")
    preload_model: bool = Field(default=True)
    corellm_timeout: int = Field(default=300)

    def __init__(self, **kwargs: Any):
        # Support the old `base_url` argument but map to our custom variable
        if "base_url" in kwargs and "corellm_base_url" not in kwargs:
            kwargs["corellm_base_url"] = kwargs.pop("base_url")

        # Resolve the base URL (defaults to ENV var if present, then HF Space)
        env_base = os.environ.get("CORELLM_BASE_URL", "https://namitkumar22-corellm.hf.space")
        base = kwargs.get("corellm_base_url", env_base).rstrip("/")
        
        # Set our variables
        kwargs["corellm_base_url"] = base
        
        # ChatOpenAI needs base_url to point to the OpenAI-compatible endpoint
        kwargs["base_url"] = f"{base}/v1"

        # ChatOpenAI requires an API key, we supply a dummy one if none exists
        if "api_key" not in kwargs and "openai_api_key" not in kwargs:
            kwargs["api_key"] = "not-needed"
            
        # Extract custom fields to prevent Pydantic strict-validation issues
        corellm_base_url = kwargs.pop("corellm_base_url")
        preload_model = kwargs.pop("preload", True)
        preload_model = kwargs.pop("preload_model", preload_model)
        corellm_timeout = kwargs.pop("timeout", 300)
        corellm_timeout = kwargs.pop("corellm_timeout", corellm_timeout)

        super().__init__(**kwargs)
        
        # Manually set the custom fields after init
        self.corellm_base_url = corellm_base_url
        self.preload_model = preload_model
        self.corellm_timeout = corellm_timeout

        if self.preload_model:
            self._preload(self.model_name)

    @property
    def _llm_type(self) -> str:
        return "corellm_sdk"

    # ── Internals ─────────────────────────────────────────────────────────────

    @property
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from huggingface_hub import get_token
                token = get_token()
            except ImportError:
                pass
                
        if not token and hasattr(self, "api_key") and self.api_key:
            api_key_str = self.api_key.get_secret_value() if hasattr(self.api_key, "get_secret_value") else str(self.api_key)
            if api_key_str != "not-needed":
                token = api_key_str
                
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        return headers

    def _post(self, path: str, body: dict) -> dict:
        with httpx.Client(timeout=self.corellm_timeout) as client:
            r = client.post(f"{self.corellm_base_url}{path}", json=body, headers=self._headers)
            r.raise_for_status()
            return r.json()

    def _get(self, path: str) -> dict:
        with httpx.Client(timeout=self.corellm_timeout) as client:
            r = client.get(f"{self.corellm_base_url}{path}", headers=self._headers)
            r.raise_for_status()
            return r.json()

    def _preload(self, model: str):
        print(f"[CoreLLM SDK] Pre-warming '{model}' on server...")
        try:
            self._post("/api/load", {"model": model})
            print(f"[CoreLLM SDK] ✓ '{model}' is ready.")
        except Exception as e:
            print(f"[CoreLLM SDK] Failed to pre-warm model: {e}")

    # ── Model control ─────────────────────────────────────────────────────────

    def switch(self, new_model: str) -> "CoreLLMChat":
        """
        Switch the active model on the server and update this instance.
        Previous model is unloaded from RAM automatically.
        """
        print(f"[CoreLLM SDK] Switching '{self.model_name}' → '{new_model}'...")
        self._post("/api/switch", {"model": new_model})
        self.model_name = new_model
        print(f"[CoreLLM SDK] ✓ Active model is now '{new_model}'.")
        return self

    def unload(self) -> dict:
        """Release the current model from server RAM."""
        result = self._post("/api/unload", {"model": self.model_name})
        print(f"[CoreLLM SDK] '{self.model_name}' unloaded from memory.")
        return result
        
    def list_models(self) -> list[str]:
        """Return all models available on the server."""
        return self._get("/api/models").get("models", [])

    def status(self) -> dict:
        """Return server health and active model info."""
        return self._get("/")

    # ── Additional Inference Endpoints ────────────────────────────────────────

    def raw_chat(self, messages: list[dict], **kwargs) -> str:
        """
        Multi-turn chat using the Ollama /api/chat endpoint.
        """
        body = {"model": self.model_name, "messages": messages, "stream": False, **kwargs}
        result = self._post("/api/chat", body)
        return result.get("message", {}).get("content", "")

    def raw_generate(self, prompt: str, **kwargs) -> str:
        """
        Raw text completion using the Ollama /api/generate endpoint.
        """
        body = {"model": self.model_name, "prompt": prompt, "stream": False, **kwargs}
        result = self._post("/api/generate", body)
        return result.get("response", "")

    def openai_chat(self, messages: list[dict], **kwargs) -> str:
        """
        OpenAI-compatible /v1/chat/completions endpoint.
        Compatible with any tool expecting the OpenAI response format.
        
        Returns the raw string content.
        """
        body = {"model": self.model_name, "messages": messages, **kwargs}
        result = self._post("/v1/chat/completions", body)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return str(result)
