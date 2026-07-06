# CoreLLM SDK

**The simplest way to use powerful LLMs — no API key, no setup.**

```bash
pip install corellm-sdk
```

---

## 🤖 Available Models

All models run on a single endpoint (4× A10G, 96 GB VRAM).
Models are loaded into VRAM **on demand** — only the model you call is active.

| Model | Capabilities | Context |
|-------|-------------|---------|
| `qwen3.6:35b` | coding, text, tools, thinking | 256k |
| `qwen3-vl:32b` | vision, text, tools, thinking | 256k |
| `nemotron3:33b` | audio, text, tools, vision, thinking | — |
| `lfm2.5-thinking:1.2b` | ultra fast, tools, thinking | 32k |
| `qwen3-embedding:8b` | embedding | — |

---

## 🚀 Quickstart

```python
from corellm_sdk import CoreLLMChat, ENDPOINT

llm = CoreLLMChat(
    model="qwen3.6:35b",
    base_url=ENDPOINT,
)
```

---

## 🧩 LangChain & LangGraph

Full drop-in replacement for `ChatOpenAI` — agents, `bind_tools`, LCEL chains, streaming all work.

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
embed_llm = CoreLLMChat(model="qwen3-embedding:8b")
vector = embed_llm.embed("The quick brown fox")
print(len(vector), "dimensions")
```

---

## 🔄 Model Switching

```python
# Switch from qwen3.6 to vision model on the fly
# Previous model is automatically unloaded from VRAM
llm.switch("qwen3-vl:32b")
print(llm.raw_generate("Describe this image."))

# Switch to ultra-fast model
llm.switch("lfm2.5-thinking:1.2b")
```

---

## 🛠 Server Utilities

```python
# Check server status and which model is currently in VRAM
print(llm.status())

# List all models available on disk
print(llm.list_models())

# Manually release current model from VRAM
llm.unload()
```

---

## Environment Variable

```bash
# Override the default endpoint for all CoreLLMChat instances
export CORELLM_BASE_URL=https://your-custom-endpoint.modal.run
```
