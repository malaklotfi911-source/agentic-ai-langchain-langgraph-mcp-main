"""Orchestrateur LangGraph du SAV.

START -> orchestrateur -> agent_documentaire | agent_garantie -> orchestrateur -> END
L'orchestrateur analyse la demande, choisit UN agent specialise, et termine des qu'une reponse finale existe.
"""
import re
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from agents import creer_noeud_documentaire, creer_noeud_garantie, derniere_question
from config import get_model


class SAVState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    image_ref: str
    categorie: str
    agent_utilise: str
    reponse_finale: str


class Categorie(BaseModel):
    categorie: Literal["documentation", "garantie"]


def classer(question: str, image_ref: str) -> str:
    q = question.lower()
    if image_ref:                                                        # photo jointe -> depannage
        return "documentation"
    if re.search(r"garantie|numero de serie|numéro de série|serie", q):   # regle deterministe
        return "garantie"
    try:                                                                  # sinon le LLM tranche
        return get_model().with_structured_output(Categorie).invoke(
            "Classe cette demande de service apres-vente : 'documentation' (depannage, question technique) "
            f"ou 'garantie' (verification de garantie).\nDemande : {question}"
        ).categorie
    except Exception:
        return "documentation"


def orchestrateur(state: SAVState):
    if state.get("reponse_finale"):
        return {}                                   # rien a faire : le routage va terminer
    categorie = classer(derniere_question(state["messages"]), state.get("image_ref", ""))
    print(f"[orchestrateur] categorie = {categorie}")
    return {"categorie": categorie}


def router(state: SAVState):
    if state.get("reponse_finale"):
        return END
    return "agent_documentaire" if state["categorie"] == "documentation" else "agent_garantie"


async def build_graph():
    builder = StateGraph(SAVState)
    builder.add_node("orchestrateur", orchestrateur)
    builder.add_node("agent_documentaire", await creer_noeud_documentaire())
    builder.add_node("agent_garantie", creer_noeud_garantie())

    builder.add_edge(START, "orchestrateur")
    builder.add_conditional_edges("orchestrateur", router, ["agent_documentaire", "agent_garantie", END])
    builder.add_edge("agent_documentaire", "orchestrateur")   # l'agent rend la main a l'orchestrateur
    builder.add_edge("agent_garantie", "orchestrateur")
    return builder.compile()
