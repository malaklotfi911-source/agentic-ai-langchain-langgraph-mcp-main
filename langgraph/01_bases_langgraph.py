"""B.1 - Les concepts de base de LangGraph : State, nodes, edges, conditional edges, reducers, memoire, tools.

Utilisation :
    python 01_bases_langgraph.py        # execute les 6 sections
    python 01_bases_langgraph.py 3      # execute seulement la section 3
"""
import operator
import sys
from typing import Annotated, Literal, TypedDict

from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import get_model


# ---------- 1. State, nodes, edges ----------
class EtatTexte(TypedDict):
    texte: str
    longueur: int


def nettoyer(state: EtatTexte):
    return {"texte": state["texte"].strip().lower()}


def compter(state: EtatTexte):
    return {"longueur": len(state["texte"])}


def section_1():
    print("\n##### 1. State, nodes et edges #####")
    builder = StateGraph(EtatTexte)
    builder.add_node("nettoyer", nettoyer)
    builder.add_node("compter", compter)
    builder.add_edge(START, "nettoyer")
    builder.add_edge("nettoyer", "compter")
    builder.add_edge("compter", END)
    graph = builder.compile()

    print("Resultat :", graph.invoke({"texte": "  Bonjour LangGraph  "}))
    print("\nSchema du graphe (Mermaid, a coller sur mermaid.live) :")
    print(graph.get_graph().draw_mermaid())


# ---------- 2. Conditional edges ----------
class EtatNombre(TypedDict):
    nombre: int
    message: str


def routeur(state: EtatNombre) -> Literal["pair", "impair"]:
    return "pair" if state["nombre"] % 2 == 0 else "impair"


def section_2():
    print("\n##### 2. Conditional edges (routage) #####")
    builder = StateGraph(EtatNombre)
    builder.add_node("noeud_pair", lambda s: {"message": f"{s['nombre']} est pair"})
    builder.add_node("noeud_impair", lambda s: {"message": f"{s['nombre']} est impair"})
    builder.add_conditional_edges(START, routeur, {"pair": "noeud_pair", "impair": "noeud_impair"})
    builder.add_edge("noeud_pair", END)
    builder.add_edge("noeud_impair", END)
    graph = builder.compile()

    for n in (4, 7):
        print(graph.invoke({"nombre": n}))


# ---------- 3. Reducers ----------
class EtatReducer(TypedDict):
    sans_reducer: str                              # ecrase a chaque noeud
    historique: Annotated[list[str], operator.add]  # s'accumule grace au reducer


def section_3():
    print("\n##### 3. Reducers : ecraser ou accumuler #####")
    builder = StateGraph(EtatReducer)
    builder.add_node("a", lambda s: {"sans_reducer": "ecrit par A", "historique": ["A est passe"]})
    builder.add_node("b", lambda s: {"sans_reducer": "ecrit par B", "historique": ["B est passe"]})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    print(builder.compile().invoke({"sans_reducer": "", "historique": []}))


# ---------- 4. Chatbot LLM + memoire ----------
def section_4():
    print("\n##### 4. Chatbot LLM avec MessagesState et memoire (checkpointer) #####")
    model = get_model()

    def chatbot(state: MessagesState):
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "bases-1"}}
    for q in ["Je m'appelle Mehdi.", "Comment je m'appelle ?"]:
        r = graph.invoke({"messages": [("user", q)]}, config)
        print(f"Vous : {q}")
        r["messages"][-1].pretty_print()


# ---------- 5. Agent ReAct construit a la main (tools) ----------
@tool
def calculer_prix_ttc(prix_ht: float, taux_tva: float = 20.0) -> float:
    """Calcule le prix TTC a partir du prix HT et du taux de TVA en pourcentage."""
    return round(prix_ht * (1 + taux_tva / 100), 2)


@tool
def meteo(ville: str) -> str:
    """Donne la meteo (fictive) d'une ville."""
    return {"casablanca": "22 C, ensoleille", "rabat": "20 C, nuageux"}.get(ville.lower(), "Ville inconnue")


def section_5():
    print("\n##### 5. Agent ReAct : boucle agent <-> tools #####")
    tools = [calculer_prix_ttc, meteo]
    model_tools = get_model().bind_tools(tools)

    def agent(state: MessagesState):
        return {"messages": [model_tools.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)   # tools si tool_calls, sinon END
    builder.add_edge("tools", "agent")                        # retour a l'agent apres les tools
    graph = builder.compile()

    print(graph.get_graph().draw_mermaid())
    r = graph.invoke({"messages": [("user", "Prix TTC de 1500 HT et meteo a Rabat ?")]})
    for m in r["messages"]:
        m.pretty_print()


# ---------- 6. Streaming : voir chaque etape ----------
def section_6():
    print("\n##### 6. Streaming : suivre l'execution noeud par noeud #####")
    builder = StateGraph(EtatTexte)
    builder.add_node("nettoyer", nettoyer)
    builder.add_node("compter", compter)
    builder.add_edge(START, "nettoyer")
    builder.add_edge("nettoyer", "compter")
    builder.add_edge("compter", END)
    graph = builder.compile()
    for update in graph.stream({"texte": "  Streaming  "}, stream_mode="updates"):
        print(update)


SECTIONS = {"1": section_1, "2": section_2, "3": section_3, "4": section_4, "5": section_5, "6": section_6}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[k]()
