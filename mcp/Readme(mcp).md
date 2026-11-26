# Partie C - MCP

## Installation
    python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
    pip install -r requirements.txt
    ollama pull qwen3:8b

## 1. Lancer le serveur
    python server.py            # endpoint : http://127.0.0.1:8000/mcp

## 2. Tester avec l'Inspector
    npx @modelcontextprotocol/inspector
Dans l'interface : Transport Type = Streamable HTTP, URL = http://127.0.0.1:8000/mcp, Connect,
puis onglet Tools > List Tools > exécuter chaque tool (captures d'écran pour le rapport).

## 3. Agent LangChain
    python client_agent.py

## 4. Agent n8n
1. Workflow : Chat Trigger -> AI Agent (modèle Ollama Chat Model, base URL http://host.docker.internal:11434 si n8n est dans Docker).
2. Sur l'AI Agent, ajouter le tool "MCP Client Tool" :
   - Endpoint : http://host.docker.internal:8000/mcp (Docker) ou http://127.0.0.1:8000/mcp (n8n natif)
   - Server Transport : HTTP Streamable
   - Tools to Include : All
3. Si n8n est dans Docker : dans server.py mettre host="0.0.0.0". Si le serveur répond 421 / "Invalid Host header",
   configurer TransportSecuritySettings(allowed_hosts=[...]) dans FastMCP.