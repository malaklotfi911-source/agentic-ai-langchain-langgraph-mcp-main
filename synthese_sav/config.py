from langchain_ollama import ChatOllama

MODEL_NAME = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"            # mets la meme valeur que dans la partie B
MCP_URL = "http://127.0.0.1:8001/mcp"       # serveur MCP rag_multimodal


def get_model(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=MODEL_NAME, temperature=temperature, num_ctx=4096)
