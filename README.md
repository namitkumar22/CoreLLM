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

# Heavy model (nemotron-super)
llm = CoreLLMChat(
    model="nemotron-3-super:120b",
    base_url=HEAVY_ENDPOINT,
)

# Light / fast model
fast_llm = CoreLLMChat(
    model="lfm2.5-thinking:1.2b",
    base_url=LIGHT_ENDPOINT,
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
embed_llm = CoreLLMChat(model="qwen3-embedding:8b", base_url=LIGHT_ENDPOINT)
vector = embed_llm.embed("Some text to embed")
```

---

## 🔄 Model Switching

```python
llm.switch("gemma4:31b")
print(llm.raw_generate("Describe what you see."))
```