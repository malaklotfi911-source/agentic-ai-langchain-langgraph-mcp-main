"""Agents specialises : agent_documentaire (outil MCP distant) et agent_garantie (outil local)."""
import re
from datetime import timedelta

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MCP_URL, get_model
from tools import consulter_garantie


def derniere_question(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def texte_de(contenu) -> str:
    """Les outils MCP renvoient parfois une liste de blocs : on en extrait le texte."""
    if isinstance(contenu, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in contenu).strip()
    return str(contenu).strip()


async def creer_noeud_documentaire():
    client = MultiServerMCPClient({
        "rag": {
            "url": MCP_URL,
            "transport": "streamable_http",
            "timeout": timedelta(seconds=600),            # le RAG local peut etre lent
            "sse_read_timeout": timedelta(seconds=600),
        }
    })
    outils = await client.get_tools()
    rag_tool = next(t for t in outils if t.name == "rag_multimodal")

    async def agent_documentaire(state):
        question = derniere_question(state["messages"])
        image_ref = state.get("image_ref", "")
        agent = create_agent(
            get_model(), [rag_tool],
            system_prompt=(
                "Tu es l'agent documentaire du SAV. Pour toute question technique, appelle l'outil rag_multimodal "
                f"avec la question du client{' et image_ref=' + image_ref if image_ref else ''}."
            ),
        )
        result = await agent.ainvoke({"messages": [("user", question)]})
        retours = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        if retours:
            reponse = texte_de(retours[-1].content)       # reponse sourcee produite par le RAG
        else:
            print("[agent_documentaire] garde-fou : appel force de rag_multimodal")
            reponse = texte_de(await rag_tool.ainvoke({"question": question, "image_ref": image_ref}))
        return {"messages": [AIMessage(reponse)], "reponse_finale": reponse, "agent_utilise": "agent_documentaire"}

    return agent_documentaire


def creer_noeud_garantie():
    async def agent_garantie(state):
        question = derniere_question(state["messages"])
        agent = create_agent(
            get_model(), [consulter_garantie],
            system_prompt=(
                "Tu es l'agent garantie du SAV. Appelle consulter_garantie avec le numero de serie du client, "
                "puis explique en francais si l'appareil est sous garantie, avec la date de fin."
            ),
        )
        result = await agent.ainvoke({"messages": [("user", question)]})
        retours = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        if retours:
            reponse = str(result["messages"][-1].content)
        else:
            m = re.search(r"[A-Za-z]{2,4}\d{1,3}-\d{3,5}", question)
            if not m:
                reponse = "Merci de me communiquer le numero de serie de votre appareil."
            else:
                print("[agent_garantie] garde-fou : appel force de consulter_garantie")
                donnees = consulter_garantie.invoke({"numero_serie": m.group(0)})
                reponse = str(get_model().invoke(
                    f"Reponds en francais, en une ou deux phrases, a la question du client : {question}\n"
                    f"Donnees du registre : {donnees}"
                ).content)
        return {"messages": [AIMessage(reponse)], "reponse_finale": reponse, "agent_utilise": "agent_garantie"}

    return agent_garantie
