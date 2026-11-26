"""Agent LangChain qui utilise le serveur MCP via streamable-http.
Prérequis : serveur lancé (python server.py) et `ollama pull qwen3:8b`.
"""
import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


async def main():
    client = MultiServerMCPClient(
        {
            "sav": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    print("Tools MCP chargés :", [t.name for t in tools])

    model = ChatOllama(model="qwen2.5:7b", temperature=0)
    agent = create_agent(
        model,
        tools,
        system_prompt="Tu es un assistant commercial. Utilise les tools pour répondre, en français.",
    )

    question = "Quel est le prix TTC de l'imprimante PX-200 et est-elle en stock ?"
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    for m in result["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())