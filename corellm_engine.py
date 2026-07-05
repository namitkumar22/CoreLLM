"""
CoreLLM Engine Python Client SDK
================================
A LangChain-native chat model client for your CoreLLM Hugging Face Space.

Usage
-----
    from corellm_engine import CoreLLMChat
    from langchain_core.messages import HumanMessage
    
    llm = CoreLLMChat(
        model="gemma4:e4b"
    )
    
    # LangChain usage
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
from langchain_core.language_models.chat_models import BaseChatModel
# pyrefly: ignore [missing-import]
from langchain_core.messages import BaseMessage, AIMessage
# pyrefly: ignore [missing-import]
from langchain_core.outputs import ChatGeneration, ChatResult
# pyrefly: ignore [missing-import]
from langchain_core.callbacks import CallbackManagerForLLMRun


class CoreLLMChat(BaseChatModel):
    """
    LangChain-compatible chat model backed by your CoreLLM HF Space.

    Drop-in replacement for ChatOpenAI — use it in any LangChain
    chain or LangGraph graph. Also includes raw OpenAI compatibility methods.
    
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

    model: str
    base_url: str = "https://namitkumar22-corellm.hf.space"
    preload: bool = True
    timeout: int = 300

    def model_post_init(self, __context: Any) -> None:
        """Called automatically after __init__ by pydantic v2."""
        # Need to re-assign properly if missing
        base = self.base_url or os.environ.get("CORELLM_BASE_URL", "https://namitkumar22-corellm.hf.space")
        self.base_url = base.rstrip("/")

        if self.preload:
            self._preload(self.model)

    @property
    def _llm_type(self) -> str:
        return "corellm_engine"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "base_url": self.base_url}

    # ── Internals ─────────────────────────────────────────────────────────────

    @property
    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

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
        print(f"[CoreLLM Engine] Pre-warming '{model}' on server...")
        try:
            self._post("/api/load", {"model": model})
            print(f"[CoreLLM Engine] ✓ '{model}' is ready.")
        except Exception as e:
            print(f"[CoreLLM Engine] Failed to pre-warm model: {e}")

    # ── Model control ─────────────────────────────────────────────────────────

    def switch(self, new_model: str) -> "CoreLLMChat":
        """
        Switch the active model on the server and update this instance.
        Previous model is unloaded from RAM automatically.
        """
        print(f"[CoreLLM Engine] Switching '{self.model}' → '{new_model}'...")
        self._post("/api/switch", {"model": new_model})
        self.model = new_model
        print(f"[CoreLLM Engine] ✓ Active model is now '{new_model}'.")
        return self

    def unload(self) -> dict:
        """Release the current model from server RAM."""
        result = self._post("/api/unload", {"model": self.model})
        print(f"[CoreLLM Engine] '{self.model}' unloaded from memory.")
        return result
        
    def list_models(self) -> list[str]:
        """Return all models available on the server."""
        return self._get("/api/models").get("models", [])

    def status(self) -> dict:
        """Return server health and active model info."""
        return self._get("/")

    # ── LangChain Core ────────────────────────────────────────────────────────

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "function": "function",
            "tool": "tool",
        }
        result = []
        for m in messages:
            role = role_map.get(m.type, m.type)
            result.append({"role": role, "content": str(m.content)})
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg_dicts = self._convert_messages(messages)
        extra = {}
        if stop:
            extra["stop"] = stop

        content = self.raw_chat(msg_dicts, **extra, **kwargs)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ── Additional Inference Endpoints ────────────────────────────────────────

    def raw_chat(self, messages: list[dict], **kwargs) -> str:
        """
        Multi-turn chat using the Ollama /api/chat endpoint.
        """
        body = {"model": self.model, "messages": messages, "stream": False, **kwargs}
        result = self._post("/api/chat", body)
        return result.get("message", {}).get("content", "")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Raw text completion using the Ollama /api/generate endpoint.
        """
        body = {"model": self.model, "prompt": prompt, "stream": False, **kwargs}
        result = self._post("/api/generate", body)
        return result.get("response", "")

    def openai_chat(self, messages: list[dict], **kwargs) -> str:
        """
        OpenAI-compatible /v1/chat/completions endpoint.
        Compatible with any tool expecting the OpenAI response format.
        
        Returns the raw string content.
        """
        body = {"model": self.model, "messages": messages, **kwargs}
        result = self._post("/v1/chat/completions", body)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return str(result)
