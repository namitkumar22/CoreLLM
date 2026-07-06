---
title: CoreLLM SDK
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

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

```python
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Direct invocation
response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)

# LCEL chain
chain = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
]) | llm

print(chain.invoke({"question": "What is Python?"}).content)
```

---

## ⚡ Raw APIs

```python
print(llm.raw_generate("Explain quantum physics in one sentence."))
print(llm.raw_chat([{"role": "user", "content": "Who are you?"}]))
```

---

## 🔢 Embeddings

```python
embed_llm = CoreLLMChat(model="qwen3-embedding:8b")
vector = embed_llm.embed("Some text to embed")
```

---

## 🔄 Model Switching

Previous model is automatically unloaded from VRAM on switch.

```python
llm.switch("qwen3-vl:32b")
llm.switch("lfm2.5-thinking:1.2b")
```