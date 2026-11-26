# Partie D - Synthese SAV (orchestrateur LangGraph + MCP)

## Architecture
    client -> orchestrateur (LangGraph) -> agent_documentaire --(client MCP)--> serveur MCP: rag_multimodal
                                                                                 -> agent RAG (passages + figures)
                                        -> agent_garantie --(outil local)--> consulter_garantie (registre SQLite, lecture seule)
    chaque agent rend la main a l'orchestrateur, qui termine des qu'une reponse finale existe.

## Lancement
    pip install -r requirements.txt
    python init_data.py                 # donnees de demonstration (manuel, figures, photo, registre)
    python registre.py PX200-0042       # test du registre seul
    python mcp_server_rag.py            # terminal 1 : serveur MCP (http://127.0.0.1:8001/mcp)
    python tests.py                     # terminal 2 : les deux exemples du sujet

Le serveur MCP peut aussi etre teste avec `npx @modelcontextprotocol/inspector` (Streamable HTTP, URL ci-dessus).

## Limites de la version legere
- Les figures sont retrouvees via leur legende/description (embeddings de texte), pas via CLIP.
- La photo jointe est decrite par un catalogue (data/photos_clients.json) qui simule l'analyse d'image.
  Pour une vraie analyse : modele de vision Ollama (qwen2.5vl) ou CLIP dans `rag_agent.decrire_photo`.
