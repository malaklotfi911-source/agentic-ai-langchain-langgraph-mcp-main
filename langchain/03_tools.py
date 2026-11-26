"""A.3 et A.4 - Tools personnalises et tools predefinis (DuckDuckGo, Tavily, PythonREPLTool)."""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from config import get_model

load_dotenv()


def ask(agent, question, thread="t1"):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": thread}},
    )
    for m in result["messages"]:
        m.pretty_print()
    return result


def reponse_vide(result):
    """Vrai si le modele a renvoye un message vide sans appeler de tool (raté courant des petits modeles)."""
    last = result["messages"][-1]
    return not str(last.content).strip() and not getattr(last, "tool_calls", None)


def safe_ask(titre, agent, question, thread):
    """Lance une question sans que l'echec d'un tool n'arrete le script ; relance si la reponse est vide."""
    print(f"\n##### {titre} #####")
    try:
        for essai in range(3):
            q = question if essai == 0 else question + " (Appelle le tool, puis donne la reponse finale.)"
            result = ask(agent, q, f"{thread}-{essai}")
            if not reponse_vide(result):
                return
            print(f"[reponse vide, nouvel essai {essai + 1}/2]")
        print("[ECHEC] Le modele n'a pas produit de reponse (limite du petit modele).")
    except Exception as e:
        print(f"[ECHEC] {type(e).__name__}: {str(e)[:300]}")


@wrap_tool_call
def gestion_erreurs_tools(request, handler):
    """Gere les erreurs des tools et les sorties vides du REPL Python (code sans print)."""
    try:
        result = handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Erreur du tool : {e}. Rappelle le tool avec les bons arguments.",
            tool_call_id=request.tool_call["id"],
        )
    # Le REPL ne renvoie que ce qui est affiche avec print() : sortie vide = code sans print
    if (
        request.tool_call["name"] == "Python_REPL"
        and isinstance(result, ToolMessage)
        and not str(result.content).strip()
    ):
        return ToolMessage(
            content="Sortie vide : le code n'a rien affiche. Rappelle Python_REPL avec du code qui "
                    "affiche le resultat avec print(...).",
            tool_call_id=request.tool_call["id"],
        )
    return result


def build(tools, prompt):
    return create_agent(
        get_model(), tools, system_prompt=prompt,
        middleware=[gestion_erreurs_tools], checkpointer=InMemorySaver(),
    )


# ---------- 1. Tools personnalises ----------
@tool
def calculer_prix_ttc(prix_ht: float, taux_tva: float = 20.0) -> float:
    """Calcule le prix TTC a partir du prix HT et du taux de TVA en pourcentage."""
    return round(prix_ht * (1 + taux_tva / 100), 2)


@tool
def meteo(ville: str) -> str:
    """Donne la meteo (fictive) d'une ville."""
    data = {"casablanca": "22 C, ensoleille", "rabat": "20 C, nuageux"}
    return data.get(ville.lower(), "Ville inconnue")


print("\n##### 1. Tools personnalises #####")
agent = build([calculer_prix_ttc, meteo], "Assistant en francais. Utilise les tools quand c'est utile.")
ask(agent, "Quel est le prix TTC de 1500 DH HT, et quel temps fait-il a Casablanca ?", "custom")

# ---------- 2. DuckDuckGo (sans cle API) ----------
BACKENDS = ["duckduckgo", "bing", "brave", "yahoo", "auto"]


@tool
def duckduckgo_search(query: str) -> str:
    """Recherche sur le web (DuckDuckGo) et retourne les meilleurs resultats avec titre, lien et extrait."""
    from ddgs import DDGS

    derniere_erreur = None
    for backend in BACKENDS:   # certains moteurs peuvent etre bloques selon la connexion : on essaie les suivants
        try:
            resultats = DDGS(timeout=8).text(query, region="fr-fr", max_results=3, backend=backend)
            if resultats:
                return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in resultats)
        except Exception as e:
            derniere_erreur = e
    return f"Recherche indisponible : {derniere_erreur}"


ddg = duckduckgo_search
agent_ddg = build([ddg], "Assistant en francais. Utilise la recherche web pour les infos recentes.")
safe_ask("2. DuckDuckGo", agent_ddg, "Quelle est la derniere version stable de Python ?", "ddg")

# ---------- 3. Tavily (cle TAVILY_API_KEY dans .env) ----------
tavily = None
if os.getenv("TAVILY_API_KEY"):
    from langchain_tavily import TavilySearch

    tavily = TavilySearch(max_results=3)
    agent_tav = build([tavily], "Assistant en francais. Utilise la recherche web pour les infos recentes.")
    safe_ask("3. Tavily", agent_tav, "Quelles sont les actualites recentes sur LangChain ?", "tavily")
else:
    print("\n##### 3. Tavily #####")
    print("TAVILY_API_KEY absente : cree un fichier .env (voir .env.example). Etape ignoree.")

# ---------- 4. PythonREPLTool ----------
from langchain_experimental.tools import PythonREPLTool

repl = PythonREPLTool()
PROMPT_REPL = (
    "Tu es un assistant en francais. Pour tout calcul, tu DOIS appeler le tool Python_REPL. "
    "Ce tool prend UN SEUL argument, nomme 'query', qui contient le code Python complet. "
    "Le code doit afficher le resultat avec print(). "
    "N'ecris jamais le code dans ta reponse sans l'executer. "
    "Apres avoir recu le resultat du tool, donne la reponse finale en une phrase."
)
agent_py = build([repl], PROMPT_REPL)
safe_ask("4. PythonREPLTool", agent_py,
         "Utilise le tool Python_REPL pour calculer la factorielle de 20 et la somme des carres des entiers de 1 a 10.", "repl")

# ---------- 5. Tous les tools ensemble ----------
all_tools = [calculer_prix_ttc, meteo, ddg, repl] + ([tavily] if tavily else [])
agent_all = build(
    all_tools,
    "Assistant en francais. Utilise un tool pour chaque question : "
    "pour un calcul, appelle Python_REPL (argument 'query' = code Python avec print()) ; "
    "pour une info recente, utilise la recherche web ; pour un prix TTC, utilise calculer_prix_ttc. "
    "Reponds seulement apres avoir recu le resultat du tool.",
)
safe_ask("5. Tous les tools ensemble", agent_all, "Calcule 37 * 91 avec Python.", "all")