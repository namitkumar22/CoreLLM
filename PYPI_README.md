# CoreLLM SDK

**The simplest way to use powerful LLMs — no API key, no setup.**

```bash
pip install corellm-sdk
```

---

## 🤖 Available Models

| Model | Tier | Capabilities | Context |
|-------|------|-------------|---------|
| `nemotron-3-super:120b` | Heavy | text, tools, thinking | 256k |
| `qwen3-vl:32b` | Heavy | text, vision, tools, thinking | 256k |
| `lfm2.5-thinking:1.2b` | Light | ultra fast, tools, thinking | 32k |
| `qwen3-embedding:8b` | Light | embedding | — |

---

## 🚀 Quickstart

```python
from corellm_sdk import CoreLLMChat, HEAVY_ENDPOINT, LIGHT_ENDPOINT

# Heavy model (nemotron-super on H200)
llm = CoreLLMChat(
    model="nemotron-3-super:120b",
    base_url=HEAVY_ENDPOINT,
)

# Light / fast model (A10G)
fast_llm = CoreLLMChat(
    model="lfm2.5-thinking:1.2b",
    base_url=LIGHT_ENDPOINT,
)
```

---

## 🧩 LangChain & LangGraph

Full drop-in replacement for `ChatOpenAI` — agents, `bind_tools`, LCEL, streaming all work out of the box.

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Direct invocation
response = llm.invoke([HumanMessage(content="Explain transformers in 2 sentences.")])
print(response.content)

# LCEL chain
chain = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("human", "{question}"),
]) | llm

print(chain.invoke({"question": "What is Python?"}).content)

# Tool binding (LangGraph agents)
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"

agent_llm = llm.bind_tools([get_weather])
result = agent_llm.invoke([HumanMessage(content="What's the weather in London?")])
print(result.tool_calls)
```

---

## ⚡ Raw APIs

```python
# Simple prompt completion
print(llm.raw_generate("Write a haiku about the ocean."))

# Multi-turn chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the capital of France?"},
]
print(llm.raw_chat(messages))

# OpenAI-format response
print(llm.openai_chat([{"role": "user", "content": "Hello!"}]))
```

---

## 🔢 Embeddings

```python
from corellm_sdk import CoreLLMChat, LIGHT_ENDPOINT

embed_llm = CoreLLMChat(
    model="qwen3-embedding:8b",
    base_url=LIGHT_ENDPOINT,
)

vector = embed_llm.embed("The quick brown fox")
print(len(vector), "dimensions")
```

---

## 🔄 Model Switching

```python
# Switch from nemotron-super to gemma4:31b on the fly
llm.switch("gemma4:31b")
print(llm.raw_generate("Describe this image."))

# Switch back
llm.switch("nemotron-super")
```

---

## 🛠 Server Utilities

```python
# Check server status
print(llm.status())

# List all models available on this endpoint
print(llm.list_models())

# Manually release a model from VRAM
llm.unload()
```

---

## Environment Variable

```bash
# Override the default base URL for all CoreLLMChat instances
export CORELLM_BASE_URL=https://your-custom-endpoint.modal.run
```
