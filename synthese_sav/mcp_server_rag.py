"""Serveur MCP (streamable-http) qui publie l'outil rag_multimodal.

Lancement : python mcp_server_rag.py   ->   http://127.0.0.1:8001/mcp
"""
from mcp.server.fastmcp import FastMCP

from rag_agent import repondre

mcp = FastMCP("rag-multimodal", host="127.0.0.1", port=8001)


@mcp.tool()
def rag_multimodal(question: str, image_ref: str = "") -> str:
    """Repond a une question sur les manuels de l'imprimante PX-200 en cherchant des passages et des figures, avec sources.
    image_ref (optionnel) : reference d'une photo jointe par le client, ex. img_px200_01."""
    return repondre(question, image_ref)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
