# CoreLLM SDK

The simplest way to use LLMs.

## 📦 Install from PyPI

```bash
pip install corellm-sdk
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

The new **CoreLLMChat** class wraps everything into a single, cohesive, LangChain-compatible chat model.

```python
from corellm_sdk import CoreLLMChat

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

## 🌐 Standard OpenAI SDK Integration

Since CoreLLM is fully OpenAI-compatible, you can also use the standard LangChain OpenAI classes directly by simply pointing the `base_url` to your Hugging Face Space.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="gemma4:e4b",
    base_url="https://namitkumar22-corellm.hf.space/v1",
    api_key="your-api-key"
)

response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)
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
