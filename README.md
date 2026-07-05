---
title: CoreLLM Engine
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# CoreLLM Engine

A fully-featured Python client and Hugging Face Space for running LLMs via Ollama — with native LangChain & LangGraph support. 

`corellm-engine` acts as an all-in-one unified model interface!

## 📦 Install from PyPI

```bash
# Minimal installation (just the client)
pip install corellm-engine

# With LangChain support
pip install "corellm-engine[langchain]"

# With LangChain + LangGraph support
pip install "corellm-engine[all]"
```

## 🤖 Available Models

The following models are available on the server. **Do not use any other model names.**
- `"gemma4:e4b"` - text, vision, tools, thinking, audio, context=128k
- `"devstral:24b"` - text, tools, context=128k
- `"cogito:14b"` - text, tools, thinking, context=128k
- `"ornith:9b"` - Text, thinking, tools, context=256k
- `"lfm2.5-thinking:1.2b"` - ultra fast, tools, thinking, context=32k
- `"qwen3-embedding:8b"` - embedding
- `"robit/ornith-vision:9b"` - vision, tools, thinking

## 🚀 Quickstart

The new **CoreLLMChat** class wraps everything into a single, cohesive, Langchain-compatible chat model that also handles normal chat generation, raw completion, and OpenAI compatibility.

```python
from corellm_engine import CoreLLMChat

# Initialize the engine
llm = CoreLLMChat(
    model="gemma4:e4b"
)
```

## 🧩 LangChain & LangGraph Support

Use it seamlessly with your existing LangChain workflows:

```python
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Direct usage
response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)

# With Chains
chain = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
]) | llm

print(chain.invoke({"question": "What is Python?"}).content)
```

## 💬 OpenAI Compatibility (`openai_chat`)

Have existing code using OpenAI structures? Just use the OpenAI method out of the box!

```python
messages = [
    {"role": "system", "content": "You are a witty assistant."},
    {"role": "user", "content": "Tell me a joke."}
]

# Calls the /v1/chat/completions endpoint just like OpenAI
response = llm.openai_chat(messages, temperature=0.7)
print(response)
```

## 🛠 Raw APIs (`raw_chat` & `generate`)

If you want simpler formats:

```python
# Raw Prompt Completion
print(llm.generate("Explain quantum physics in 1 sentence."))

# Standard Dict Chat
messages = [{"role": "user", "content": "Who are you?"}]
print(llm.raw_chat(messages))
```

## 🔄 Dynamic Model Switching
Switch models on the fly! The backend dynamically handles memory constraints and load transitions.

```python
# Switch to another allowed model on your server!
llm.switch("devstral:24b")

print(llm.generate("Hello from Devstral!"))
```