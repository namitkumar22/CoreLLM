# CoreLLM

A Python client and Hugging Face Space for running LLMs via Ollama — with LangChain & LangGraph support.

## Install

```bash
# Minimal (just the client)
pip install git+https://github.com/namitkumar22/CoreLLM.git

# With LangChain support
pip install "git+https://github.com/namitkumar22/CoreLLM.git#egg=corellm[langchain]"

# With LangChain + LangGraph
pip install "git+https://github.com/namitkumar22/CoreLLM.git#egg=corellm[all]"
```

## Usage

```python
from corellm import CoreLLM, CoreLLMChat

# Plain Python client
llm = CoreLLM(
    model="qwen2.5:1.5b",
    base_url="https://namitkumar22-corellm.hf.space",
    api_key="your-api-key",
)
print(llm.generate("What is Python?"))

# LangChain / LangGraph
from langchain_core.messages import HumanMessage

chat = CoreLLMChat(
    model="qwen2.5:1.5b",
    base_url="https://namitkumar22-corellm.hf.space",
    api_key="your-api-key",
)
print(chat.invoke([HumanMessage(content="Hello!")]).content)
```