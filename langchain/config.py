# Un seul endroit pour changer de modele (qwen2.5:3b si la RAM est limitee)
from langchain_ollama import ChatOllama

MODEL_NAME = "qwen2.5:7b"


def get_model(name: str = MODEL_NAME, temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=name, temperature=temperature, num_ctx=4096)
