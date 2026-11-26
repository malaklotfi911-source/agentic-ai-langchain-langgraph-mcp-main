"""Teste les deux exemples du sujet.

Prerequis : python init_data.py  et  python mcp_server_rag.py (dans un autre terminal).
Usage : python tests.py        # les deux exemples
        python tests.py 2      # seulement l'exemple 2
"""
import asyncio
import sys

from langchain_core.messages import HumanMessage

from orchestrateur import build_graph

EXEMPLES = [
    {
        "nom": "Exemple 1 - Assistance documentaire",
        "question": "Mon imprimante PX-200 affiche un voyant clignotant. Que signifie le pictogramme sur la photo jointe ?",
        "image_ref": "img_px200_01",
        "categorie_attendue": "documentation",
        "agent_attendu": "agent_documentaire",
    },
    {
        "nom": "Exemple 2 - Verification de garantie",
        "question": "Est-ce que mon imprimante est encore sous garantie ? Son numero de serie est PX200-0042.",
        "image_ref": "",
        "categorie_attendue": "garantie",
        "agent_attendu": "agent_garantie",
    },
]


async def main():
    graph = await build_graph()
    choix = [EXEMPLES[int(a) - 1] for a in sys.argv[1:]] or EXEMPLES
    for ex in choix:
        print("\n" + "=" * 70 + f"\n{ex['nom']}\n" + "=" * 70)
        print("Question :", ex["question"], f"(photo : {ex['image_ref']})" if ex["image_ref"] else "")
        etat, parcours = {}, []
        entree = {"messages": [HumanMessage(ex["question"])], "image_ref": ex["image_ref"]}
        async for update in graph.astream(entree, stream_mode="updates"):
            for noeud, contenu in update.items():
                parcours.append(noeud)
                etat.update(contenu or {})
        print("\nParcours :", " -> ".join(parcours))
        print("\nReponse finale :\n" + etat.get("reponse_finale", "(aucune)"))
        ok_cat = etat.get("categorie") == ex["categorie_attendue"]
        ok_agent = etat.get("agent_utilise") == ex["agent_attendu"]
        ok_fin = parcours.count("orchestrateur") == 2
        print(f"\n[{'OK' if ok_cat else 'ECHEC'}] categorie = {etat.get('categorie')} (attendue : {ex['categorie_attendue']})")
        print(f"[{'OK' if ok_agent else 'ECHEC'}] agent = {etat.get('agent_utilise')} (attendu : {ex['agent_attendu']})")
        print(f"[{'OK' if ok_fin else 'ECHEC'}] l'orchestrateur reprend la main une fois puis termine")


if __name__ == "__main__":
    asyncio.run(main())
