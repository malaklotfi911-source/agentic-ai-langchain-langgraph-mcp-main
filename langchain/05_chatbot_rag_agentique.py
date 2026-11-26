"""A.6 - Chatbot base sur un RAG agentique (LangChain).

L'agent decide quand chercher dans le document, reformule ses requetes
et peut chercher plusieurs fois.

Prerequis : ollama pull nomic-embed-text
Utilisation : python 05_chatbot_rag_agentique.py chemin/vers/document.pdf
"""
import os
import sys

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

from config import get_model

# Tolere les noms de fichier avec espaces non entoures de guillemets
PDF_PATH = " ".join(sys.argv[1:]).strip('"') if len(sys.argv) > 1 else "document.pdf"
DB_DIR = "chroma_db"
# Modele d'embeddings : nomic-embed-text (~274 Mo) est le plus adapte.
# Si tu n'as pas la place de le telecharger, mets "qwen2.5:7b" (deja present, qualite de recherche moindre).
EMBED_MODEL = "nomic-embed-text"

# ---------- 1. Indexation ----------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

if os.path.exists(DB_DIR):
    print(f"Index existant charge depuis '{DB_DIR}' (supprime ce dossier pour reindexer).")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    docs = PyPDFLoader(PDF_PATH).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ---------- 2. Tool de recherche ----------
@tool
def rechercher_document(requete: str) -> str:
    """Recherche des passages pertinents dans le document. Reformule la requete si besoin et rappelle le tool si les resultats sont insuffisants."""
    resultats = retriever.invoke(requete)
    morceaux = []
    for d in resultats:
        page = d.metadata.get("page")
        etiquette = f"page {page + 1}" if isinstance(page, int) else "page ?"
        morceaux.append(f"[{etiquette}] {d.page_content}")
    return "\n\n".join(morceaux)


# ---------- 3. Agent ----------
agent = create_agent(
    get_model(),
    [rechercher_document],
    system_prompt=(
        "Tu es un assistant qui repond uniquement a partir du document. "
        "Utilise rechercher_document pour toute question sur son contenu (plusieurs fois si besoin). "
        "Cite les pages. Si l'information est absente, dis-le. Reponds en francais."
    ),
    checkpointer=InMemorySaver(),
)

# ---------- 4. Chatbot en console ----------
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "rag-1"}}
    print("Chatbot RAG agentique (tape 'quit' pour sortir)")
    while True:
        question = input("\nVous : ")
        if question.strip().lower() in {"quit", "exit", "q"}:
            break
        last = None
        for step in agent.stream({"messages": [{"role": "user", "content": question}]}, config, stream_mode="values"):
            last = step["messages"][-1]
        last.pretty_print()