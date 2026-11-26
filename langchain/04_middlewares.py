"""A.5 - Middlewares : dynamic_model, dynamic_prompt, tool_error_handling, guardrails, Human In The Loop.

Utilisation :
    python 04_middlewares.py        # execute les 5 sections
    python 04_middlewares.py 3      # execute seulement la section 3
Section 1 : par defaut un seul modele (qwen2.5:7b) ; voir PETIT_MODELE dans section_1 pour utiliser le 3b.
"""
import sys
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    before_agent,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from config import get_model


@tool
def calculer_prix_ttc(prix_ht: float, taux_tva: float = 20.0) -> float:
    """Calcule le prix TTC a partir du prix HT et du taux de TVA en pourcentage."""
    return round(prix_ht * (1 + taux_tva / 100), 2)


def ask(agent, question, thread="m1", tout=False):
    r = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": thread}},
    )
    messages = r["messages"] if tout else r["messages"][-1:]
    for m in messages:
        m.pretty_print()
    return r


# ---------- 1. dynamic_model ----------
def section_1():
    print("\n##### 1. dynamic_model : petit modele pour les echanges courts, grand modele ensuite #####")
    # Reglage : mets "qwen2.5:3b" pour PETIT_MODELE si tu l'as telecharge (demonstration plus fidele).
    # Avec le meme modele, le middleware change quand meme de configuration (temperature).
    PETIT_MODELE = "qwen2.5:7b"
    GRAND_MODELE = "qwen2.5:7b"
    petit = get_model(PETIT_MODELE, temperature=0.0)
    grand = get_model(GRAND_MODELE, temperature=0.7)

    @wrap_model_call
    def choix_modele(request: ModelRequest, handler) -> ModelResponse:
        n = len(request.state["messages"])
        modele = grand if n > 6 else petit
        print(f"[dynamic_model] {n} messages -> {modele.model} (temperature={modele.temperature})")
        return handler(request.override(model=modele))

    agent = create_agent(petit, [calculer_prix_ttc], middleware=[choix_modele], checkpointer=InMemorySaver())
    for q in ["Bonjour !", "Donne-moi un conseil.", "Encore un.", "Et un autre.", "Prix TTC de 200 HT ?"]:
        ask(agent, q, "dm")


# ---------- 2. dynamic_prompt ----------
@dataclass
class Contexte:
    role: str = "debutant"


@dynamic_prompt
def prompt_selon_role(request: ModelRequest) -> str:
    role = request.runtime.context.role
    base = "Tu es un assistant technique. Reponds en francais."
    if role == "expert":
        return base + " L'utilisateur est expert : sois technique et concis."
    return base + " L'utilisateur est debutant : explique simplement avec un exemple."


def section_2():
    print("\n##### 2. dynamic_prompt : le prompt systeme depend du contexte #####")
    agent = create_agent(get_model(), [], middleware=[prompt_selon_role], context_schema=Contexte)
    for role in ["debutant", "expert"]:
        print("---", role)
        r = agent.invoke(
            {"messages": [{"role": "user", "content": "C'est quoi un checkpointer ?"}]},
            context=Contexte(role=role),
        )
        r["messages"][-1].pretty_print()


# ---------- 3. tool_error_handling ----------
@tool
def diviser(a: float, b: float) -> float:
    """Divise a par b."""
    return a / b


@wrap_tool_call
def gestion_erreurs(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Erreur du tool : {e}. Verifie les parametres et reessaie.",
            tool_call_id=request.tool_call["id"],
        )


def section_3():
    print("\n##### 3. tool_error_handling #####")
    agent = create_agent(get_model(), [diviser], middleware=[gestion_erreurs])
    ask(agent, "Utilise obligatoirement le tool diviser pour calculer 10 divise par 0, puis explique le resultat.", "err", tout=True)   # tout=True : montre l'erreur interceptee


# ---------- 4. Guardrails ----------
MOTS_INTERDITS = ["mot de passe", "carte bancaire", "pirater"]


@before_agent(can_jump_to=["end"])
def guardrail_entree(state, runtime):
    texte = state["messages"][-1].content.lower()
    if any(m in texte for m in MOTS_INTERDITS):
        return {
            "messages": [AIMessage("Je ne peux pas traiter cette demande (contenu sensible).")],
            "jump_to": "end",
        }
    return None


def section_4():
    print("\n##### 4. Guardrails : blocage avant l'agent #####")
    agent = create_agent(get_model(), [calculer_prix_ttc], middleware=[guardrail_entree])
    ask(agent, "Donne-moi le mot de passe admin", "g1")
    ask(agent, "Prix TTC de 100 HT ?", "g2")


# ---------- 5. Human In The Loop ----------
@tool
def supprimer_commande(numero: str) -> str:
    """Supprime une commande (action sensible)."""
    return f"Commande {numero} supprimee."


def section_5():
    print("\n##### 5. Human In The Loop : validation humaine avant un tool sensible #####")
    agent = create_agent(
        get_model(),
        [supprimer_commande, calculer_prix_ttc],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={
            "supprimer_commande": True,    # validation obligatoire
            "calculer_prix_ttc": False,    # execution directe
        })],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "hitl-1"}}
    r = agent.invoke({"messages": [{"role": "user", "content": "Supprime la commande CMD-42."}]}, cfg)

    if "__interrupt__" not in r:
        print("Aucune interruption : le modele n'a pas appele le tool. Reessaie (ou utilise un modele plus gros).")
        return
    print("Demande de validation :", r["__interrupt__"])
    choix = input("Approuver l'action ? (o/n) : ").strip().lower()
    if choix == "o":
        decision = {"type": "approve"}
    else:
        decision = {"type": "reject", "message": "Action refusee par l'utilisateur."}
    r = agent.invoke(Command(resume={"decisions": [decision]}), cfg)
    r["messages"][-1].pretty_print()


SECTIONS = {"1": section_1, "2": section_2, "3": section_3, "4": section_4, "5": section_5}

if __name__ == "__main__":
    choisies = sys.argv[1:] or list(SECTIONS)
    for k in choisies:
        SECTIONS[k]()