"""
CoreLLM Python Client SDK
=========================
A clean Python interface to your CoreLLM Hugging Face Space.

All models must be pre-baked into the Space image at build time.
The client will auto-switch models on the server as needed —
you never need to call .switch() manually unless you want to
pre-warm a model before the first inference call.

Classes
-------
CoreLLM      — plain Python client (generate, chat, switch)
CoreLLMChat  — LangChain-native BaseChatModel (works in chains & LangGraph)

Usage — CoreLLM (plain Python)
-----
    from corellm import CoreLLM

    llm = CoreLLM(
        model="qwen2.5:1.5b",
        base_url="https://namitkumar22-corellm.hf.space",
        api_key="your-secret-key",
    )
    print(llm.generate("Explain gravity in one sentence."))

    llm.model = "llama3.2:3b"          # server auto-switches on next call
    print(llm.generate("Who are you?"))

Usage — CoreLLMChat (LangChain / LangGraph)
-----
    from corellm import CoreLLMChat
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = CoreLLMChat(
        model="qwen2.5:1.5b",
        base_url="https://namitkumar22-corellm.hf.space",
        api_key="your-secret-key",
    )

    # Direct call
    response = llm.invoke([HumanMessage(content="Hello!")])
    print(response.content)

    # In a LangChain chain
    from langchain_core.prompts import ChatPromptTemplate
    chain = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]) | llm
    print(chain.invoke({"question": "What is Python?"}).content)

    # In LangGraph — just pass llm as you would ChatOpenAI
"""

from __future__ import annotations

import os
import httpx
from typing import Optional, Any, List, Iterator, Mapping

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun


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


# ══════════════════════════════════════════════════════════════════════════════
# CoreLLMChat — LangChain-native wrapper
# Works directly in LangChain chains, agents, and LangGraph nodes.
# Install: pip install langchain-core
# ══════════════════════════════════════════════════════════════════════════════

class CoreLLMChat(BaseChatModel):
    """
    LangChain-compatible chat model backed by your CoreLLM HF Space.

    Drop-in replacement for ChatOpenAI — use it in any LangChain
    chain or LangGraph graph.

    Parameters
    ----------
    model : str
        The model to use (must be in server's ALLOWED_MODELS).
    base_url : str
        Your CoreLLM Space URL.
    api_key : str, optional
        Bearer token. Falls back to CORELLM_API_KEY env var.
    preload : bool
        Pre-warm the model on init (default True).
    timeout : int
        Request timeout seconds (default 300).

    Examples
    --------
    Basic usage::

        from corellm import CoreLLMChat
        from langchain_core.messages import HumanMessage

        llm = CoreLLMChat(
            model="qwen2.5:1.5b",
            base_url="https://namitkumar22-corellm.hf.space",
            api_key="sk-corellm-abc123",
        )
        response = llm.invoke([HumanMessage(content="Hello!")])
        print(response.content)

    In a chain::

        from langchain_core.prompts import ChatPromptTemplate
        chain = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{question}"),
        ]) | llm
        result = chain.invoke({"question": "What is Python?"})
        print(result.content)

    In LangGraph::

        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage
        from typing import TypedDict, List

        class State(TypedDict):
            messages: List

        def call_model(state: State):
            response = llm.invoke(state["messages"])
            return {"messages": state["messages"] + [response]}

        graph = StateGraph(State)
        graph.add_node("model", call_model)
        graph.set_entry_point("model")
        graph.add_edge("model", END)
        app = graph.compile()

        result = app.invoke({"messages": [HumanMessage(content="Hi!")]})
        print(result["messages"][-1].content)

    Switch model mid-graph::

        llm.model = "llama3.2:3b"   # server auto-switches on next invoke
    """

    # ── Pydantic fields (LangChain uses pydantic under the hood) ──────────────
    model: str
    base_url: str
    api_key: str = ""
    preload: bool = True
    timeout: int = 300

    # Internal CoreLLM client — excluded from pydantic serialization
    _client: Optional[CoreLLM] = None

    def model_post_init(self, __context: Any) -> None:
        """Called automatically after __init__ by pydantic v2."""
        self._client = CoreLLM(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key or os.environ.get("CORELLM_API_KEY", ""),
            preload=self.preload,
            timeout=self.timeout,
        )

    @property
    def _llm_type(self) -> str:
        """Identifier used by LangChain internally."""
        return "corellm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "base_url": self.base_url}

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain message objects → plain dicts for CoreLLM."""
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
            result.append({"role": role, "content": m.content})
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Core method called by LangChain on every .invoke() / .predict() call.
        Converts LangChain messages, calls CoreLLM, returns ChatResult.
        """
        # Sync model name in case user changed llm.model after init
        self._client.model = self.model

        msg_dicts = self._convert_messages(messages)
        extra = {}
        if stop:
            extra["stop"] = stop

        content = self._client.chat(msg_dicts, **extra, **kwargs)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ── Convenience: switch model and return self for chaining ─────────────────

    def switch(self, new_model: str) -> "CoreLLMChat":
        """
        Switch the active model on the server and update this instance.
        Previous model is unloaded from RAM automatically.

        Example
        -------
            llm.switch("llama3.2:3b")
            result = llm.invoke([HumanMessage(content="Hi!")])
        """
        self._client.switch(new_model)
        self.model = new_model
        return self

    def list_models(self) -> list[str]:
        """Return models available on the server."""
        return self._client.list_models()

    def status(self) -> dict:
        """Return server health and active model info."""
        return self._client.status()

    def __repr__(self) -> str:
        return f"CoreLLMChat(model={self.model!r}, base_url={self.base_url!r})"
