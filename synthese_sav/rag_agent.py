"""Agent RAG derriere l'outil MCP `rag_multimodal` : cherche des passages ET des figures, puis repond avec ses sources."""
import json
import re
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_ollama import OllamaEmbeddings

from config import EMBED_MODEL, get_model

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DB = BASE / "chroma_sav"

_stores: dict = {}


def get_stores() -> dict:
    if not _stores:
        emb = OllamaEmbeddings(model=EMBED_MODEL)
        if DB.exists():
            _stores["passages"] = Chroma(collection_name="passages", persist_directory=str(DB), embedding_function=emb)
            _stores["figures"] = Chroma(collection_name="figures", persist_directory=str(DB), embedding_function=emb)
        else:
            manuel = json.loads((DATA / "manuel_px200.json").read_text(encoding="utf-8"))
            docs_p = [
                Document(page_content=p["texte"], metadata={"id": p["id"], "page": p["page"], "section": p["section"]})
                for p in manuel["passages"]
            ]
            docs_f = [
                Document(page_content=f"{f['legende']}. {f['description']}",
                         metadata={"id": f["id"], "page": f["page"], "fichier": f["fichier"]})
                for f in manuel["figures"]
            ]
            _stores["passages"] = Chroma.from_documents(docs_p, emb, collection_name="passages", persist_directory=str(DB))
            _stores["figures"] = Chroma.from_documents(docs_f, emb, collection_name="figures", persist_directory=str(DB))
    return _stores


@tool
def chercher_passages(requete: str) -> str:
    """Cherche des passages du manuel de l'imprimante PX-200."""
    res = get_stores()["passages"].similarity_search(requete, k=3)
    return "\n\n".join(f"[manuel p.{d.metadata['page']} | {d.metadata['section']}] {d.page_content}" for d in res)


@tool
def chercher_figures(requete: str) -> str:
    """Cherche des figures (voyants, pictogrammes) du manuel de l'imprimante PX-200."""
    res = get_stores()["figures"].similarity_search(requete, k=2)
    return "\n\n".join(f"[{d.metadata['id']} | manuel p.{d.metadata['page']}] {d.page_content}" for d in res)


def decrire_photo(image_ref: str) -> str:
    """Etape 'vision' : retourne la description de la photo jointe.

    Version legere : description prise dans un catalogue (data/photos_clients.json).
    A remplacer par un modele de vision (ex. qwen2.5vl avec Ollama) ou par CLIP.
    """
    photos = json.loads((DATA / "photos_clients.json").read_text(encoding="utf-8"))
    info = photos.get(image_ref)
    return info["description"] if info else ""


SYSTEM = (
    "Tu es un agent RAG pour le service apres-vente de l'imprimante PX-200. "
    "Pour repondre, appelle chercher_figures ET chercher_passages. Reponds uniquement a partir de leurs resultats, "
    "en francais, en citant les sources entre crochets (ex. [manuel p.12], [fig_01]). "
    "Si l'information est absente, dis-le."
)


def _sources(textes: list[str]) -> str:
    tags = []
    for t in textes:
        for tag in re.findall(r"\[(?:manuel p\.\d+|fig_\d+)[^\]]*\]", t):
            if tag not in tags:
                tags.append(tag)
    return "Sources : " + " ; ".join(tags) if tags else ""


def repondre(question: str, image_ref: str = "") -> str:
    description = decrire_photo(image_ref) if image_ref else ""
    prompt = question
    if description:
        prompt += f"\n\nPhoto jointe ({image_ref}) : {description}"

    agent = create_agent(get_model(), [chercher_passages, chercher_figures], system_prompt=SYSTEM)
    result = agent.invoke({"messages": [("user", prompt)]})
    outils = [str(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]

    if outils:
        reponse = str(result["messages"][-1].content)
    else:   # garde-fou : le petit modele n'a appele aucun outil -> recherche forcee
        print("[rag] garde-fou : recherche forcee")
        requete = f"{question} {description}".strip()
        outils = [chercher_figures.invoke({"requete": requete}), chercher_passages.invoke({"requete": requete})]
        reponse = str(get_model().invoke([
            ("system", SYSTEM),
            ("user", f"{prompt}\n\nElements trouves :\n" + "\n\n".join(outils)),
        ]).content)

    src = _sources(outils)
    return f"{reponse}\n\n{src}" if src and "Sources" not in reponse else reponse
