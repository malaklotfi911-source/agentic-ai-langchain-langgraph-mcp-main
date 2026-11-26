"""Agentic RAG avec LangGraph (Ollama).

Flux :
  START -> generate_query_or_respond -> (tool call ?) -> retrieve -> grade_documents
        -> pertinent : generate_answer -> END
        -> non pertinent : rewrite_question -> generate_query_or_respond (max 2 reecritures)
        -> pas de tool call : reponse directe -> END

Ce fichier expose `graph` (pour LangGraph Studio) et `builder` (pour compiler avec memoire).
"""
import uuid
from pathlib import Path
from typing import Literal

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# ---------------- Reglages ----------------
MODEL_NAME = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"   # mets la meme valeur que dans 05_chatbot_rag_agentique.py
MAX_REECRITURES = 2
LANGUE_DOCUMENT = "anglais"   # langue du PDF ("" pour desactiver la traduction des requetes)

BASE = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE / "documents"
DB_DIR = BASE / "chroma_db_langgraph"


def get_model(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=MODEL_NAME, temperature=temperature, num_ctx=4096)


# ---------------- Indexation + retriever ----------------
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        if DB_DIR.exists():
            store = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
        else:
            pdfs = sorted(DOCS_DIR.glob("*.pdf"))
            if not pdfs:
                raise FileNotFoundError(f"Aucun PDF dans {DOCS_DIR} : copie ton document dans ce dossier.")
            docs = [d for pdf in pdfs for d in PyPDFLoader(str(pdf)).load()]
            chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
            print(f"[indexation] {len(docs)} pages -> {len(chunks)} chunks")
            store = Chroma.from_documents(chunks, embeddings, persist_directory=str(DB_DIR))
        _retriever = store.as_retriever(search_kwargs={"k": 4})
    return _retriever


def traduire_requete(requete: str) -> str:
    """Traduit la requete dans la langue du document (meilleure recherche si le PDF n'est pas en francais)."""
    if not LANGUE_DOCUMENT:
        return requete
    try:
        r = get_model().invoke(
            f"Traduis en {LANGUE_DOCUMENT} cette requete de recherche. "
            f"Donne uniquement la traduction, sans commentaire : {requete}"
        )
        return str(r.content).strip().strip('"') or requete
    except Exception:
        return requete


@tool
def rechercher_document(requete: str) -> str:
    """Recherche des passages pertinents dans le document de l'utilisateur."""
    requetes = [requete]
    traduite = traduire_requete(requete)
    if traduite.lower() != requete.lower():
        requetes.append(traduite)

    vus, resultats = set(), []
    for q in requetes:
        for d in get_retriever().invoke(q):
            if d.page_content not in vus:
                vus.add(d.page_content)
                resultats.append(d)
    resultats = resultats[:6]

    pages = [d.metadata.get("page") + 1 if isinstance(d.metadata.get("page"), int) else "?" for d in resultats]
    print(f"[recherche] requetes={requetes} -> pages {pages}")

    morceaux = []
    for d, p in zip(resultats, pages):
        morceaux.append(f"[page {p}] {d.page_content}")
    return "\n\n".join(morceaux)


# ---------------- Etat ----------------
class State(MessagesState):
    reecritures: int


def derniere_question(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


# ---------------- Noeuds ----------------
SALUTATIONS = {"bonjour", "salut", "bonsoir", "hello", "hey", "coucou", "merci", "bye", "ciao"}


def est_salutation(question: str) -> bool:
    mots = question.lower().replace("!", " ").replace("?", " ").replace(",", " ").split()
    return bool(mots) and mots[0] in SALUTATIONS and len(mots) <= 4


def generate_query_or_respond(state: State):
    """Le LLM decide : appeler l'outil de recherche ou repondre directement.

    Un petit modele local oublie souvent d'appeler l'outil : on le force par un prompt strict,
    puis par un garde-fou deterministe si besoin.
    """
    system = SystemMessage(
        "Tu es un assistant qui repond en francais a partir d'un document. "
        "Pour TOUTE question autre qu'une simple salutation ou un remerciement, tu DOIS appeler l'outil "
        "rechercher_document avant de repondre. Ne reponds jamais de memoire et ne propose jamais de chercher : "
        "cherche directement. Pour une salutation, reponds brievement sans outil."
    )
    reponse = get_model().bind_tools([rechercher_document]).invoke([system] + state["messages"])

    question = derniere_question(state["messages"])
    if not reponse.tool_calls and not est_salutation(question):
        print("[garde-fou] le modele n'a pas cherche : appel force de rechercher_document")
        reponse = AIMessage(
            content="",
            tool_calls=[{"name": "rechercher_document", "args": {"requete": question}, "id": str(uuid.uuid4())}],
        )
    return {"messages": [reponse]}


class Pertinence(BaseModel):
    score: Literal["oui", "non"] = Field(description="'oui' si les passages permettent de repondre a la question, sinon 'non'")


def grade_documents(state: State) -> Literal["generate_answer", "rewrite_question"]:
    """Arete conditionnelle : les passages recuperes sont-ils pertinents ?"""
    if state.get("reecritures", 0) >= MAX_REECRITURES:
        return "generate_answer"
    question = derniere_question(state["messages"])
    contexte = str(state["messages"][-1].content)
    prompt = (
        "Evalue si ces passages permettent de repondre a la question.\n\n"
        f"Passages :\n{contexte}\n\nQuestion : {question}\n\nReponds 'oui' ou 'non'."
    )
    try:
        verdict = get_model().with_structured_output(Pertinence).invoke(prompt)
        return "generate_answer" if verdict.score == "oui" else "rewrite_question"
    except Exception:
        return "generate_answer"   # en cas d'echec du format structure, on continue


def rewrite_question(state: State):
    question = derniere_question(state["messages"])
    reponse = get_model().invoke(
        f"Reformule cette question pour mieux chercher dans un document. Question : {question}\n"
        "Donne uniquement la question reformulee, en une phrase."
    )
    print(f"[rewrite_question] {question!r} -> {reponse.content!r}")
    return {
        "messages": [HumanMessage(content=str(reponse.content))],
        "reecritures": state.get("reecritures", 0) + 1,
    }


def generate_answer(state: State):
    question = derniere_question(state["messages"])
    contexte = str(state["messages"][-1].content)
    system = (
        "Tu reponds a la question uniquement a partir des passages ci-dessous. "
        "Cite les pages. Si l'information est absente, dis-le. Reponds en francais.\n\n"
        f"Passages :\n{contexte}"
    )
    reponse = get_model().invoke([("system", system), ("user", question)])
    return {"messages": [reponse], "reecritures": 0}


# ---------------- Graphe ----------------
builder = StateGraph(State)
builder.add_node("generate_query_or_respond", generate_query_or_respond)
builder.add_node("retrieve", ToolNode([rechercher_document]))
builder.add_node("rewrite_question", rewrite_question)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "generate_query_or_respond")
builder.add_conditional_edges("generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END})
builder.add_conditional_edges("retrieve", grade_documents)
builder.add_edge("generate_answer", END)
builder.add_edge("rewrite_question", "generate_query_or_respond")

graph = builder.compile()   # sans checkpointer : LangGraph Studio fournit le sien