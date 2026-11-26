"""Genere les donnees de demonstration : manuel PX-200 (passages + figures), photo client et registre de garanties.

Usage : python init_data.py
"""
import json
import sqlite3
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

MANUEL = {
    "produit": "PX-200",
    "passages": [
        {"id": "p12-1", "page": 12, "section": "Voyants d'etat",
         "texte": "Le voyant d'etat de la PX-200 indique l'etat de l'imprimante : vert fixe = prete ; orange clignotant = bourrage papier ; rouge fixe = cartouche vide ; bleu clignotant = connexion Wi-Fi en cours."},
        {"id": "p12-2", "page": 12, "section": "Bourrage papier",
         "texte": "Lorsque le voyant orange clignote et que le pictogramme de feuille froissee s'affiche, un bourrage papier est detecte. Eteignez l'imprimante, ouvrez le capot arriere, retirez delicatement la feuille coincee dans le sens de defilement, refermez le capot puis rallumez. Le voyant redevient vert."},
        {"id": "p13-1", "page": 13, "section": "Cartouche d'encre",
         "texte": "Un voyant rouge fixe avec le pictogramme de goutte signale que la cartouche d'encre est vide ou mal installee. Remplacez-la par une cartouche PX-INK-200 et verifiez qu'elle est bien enclenchee."},
        {"id": "p14-1", "page": 14, "section": "Connexion sans fil",
         "texte": "Un voyant bleu clignotant avec le pictogramme d'ondes indique que l'imprimante cherche le reseau Wi-Fi. Si le clignotement dure plus de deux minutes, redemarrez la box puis relancez l'appairage en maintenant le bouton Wi-Fi pendant 5 secondes."},
        {"id": "p20-1", "page": 20, "section": "Entretien",
         "texte": "Nettoyez les rouleaux d'entrainement tous les six mois avec un chiffon non pelucheux legerement humide."},
        {"id": "p21-1", "page": 21, "section": "Garantie constructeur",
         "texte": "La PX-200 est couverte par une garantie constructeur de 24 mois a compter de la date d'achat. Elle ne couvre pas les dommages lies a un usage non conforme."},
    ],
    "figures": [
        {"id": "fig_01", "page": 12, "fichier": "figures/fig_01.png",
         "legende": "Figure 3.1 - Voyant orange clignotant et pictogramme de feuille froissee",
         "description": "Le voyant d'etat orange clignote ; a cote s'affiche l'icone d'une feuille de papier froissee, signe d'un bourrage papier."},
        {"id": "fig_02", "page": 13, "fichier": "figures/fig_02.png",
         "legende": "Figure 3.2 - Voyant rouge fixe et pictogramme de goutte",
         "description": "Le voyant rouge est allume en continu avec une icone de goutte d'encre : cartouche vide ou mal installee."},
        {"id": "fig_03", "page": 14, "fichier": "figures/fig_03.png",
         "legende": "Figure 3.3 - Voyant bleu clignotant et pictogramme d'ondes",
         "description": "Le voyant bleu clignote avec une icone d'ondes radio : l'imprimante recherche le reseau Wi-Fi."},
        {"id": "fig_04", "page": 12, "fichier": "figures/fig_04.png",
         "legende": "Figure 3.4 - Voyant vert fixe",
         "description": "Le voyant vert est allume en continu avec une coche : l'imprimante est prete a imprimer."},
    ],
}

# Photos envoyees par les clients. La "description" simule l'etape d'analyse d'image (voir README).
PHOTOS = {
    "img_px200_01": {
        "fichier": "photos/img_px200_01.png",
        "description": "Photo du panneau de l'imprimante : voyant d'etat orange allume en clignotement, avec a cote l'icone d'une feuille de papier froissee.",
    }
}

REGISTRE = [
    ("PX200-0042", "Imprimante PX-200", "2025-03-15", 24),
    ("PX200-0099", "Imprimante PX-200", "2023-01-10", 24),
    ("SC10-0007", "Scanner SC-10", "2026-06-01", 12),
]


def dessiner(chemin: Path, couleur: str, symbole: str, clignote: bool):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 200), (30, 30, 36))
    d = ImageDraw.Draw(img)
    cx, cy = 90, 100
    if clignote:
        d.ellipse((cx - 52, cy - 52, cx + 52, cy + 52), outline=couleur, width=3)
    d.ellipse((cx - 36, cy - 36, cx + 36, cy + 36), fill=couleur)
    x, y = 210, 100
    blanc = (235, 235, 235)
    if symbole == "papier":
        d.polygon([(x - 28, y - 38), (x + 18, y - 38), (x + 32, y - 22), (x + 32, y + 38), (x - 28, y + 38)], outline=blanc, width=3)
        d.line([(x - 18, y - 10), (x + 6, y), (x - 14, y + 12), (x + 20, y + 24)], fill=blanc, width=3)
    elif symbole == "goutte":
        d.polygon([(x, y - 40), (x - 24, y + 5), (x + 24, y + 5)], fill=blanc)
        d.ellipse((x - 24, y - 10, x + 24, y + 38), fill=blanc)
    elif symbole == "ondes":
        for r in (16, 32, 48):
            d.arc((x - r, y - r + 20, x + r, y + r + 20), 210, 330, fill=blanc, width=4)
        d.ellipse((x - 5, y + 14, x + 5, y + 24), fill=blanc)
    elif symbole == "coche":
        d.line([(x - 30, y), (x - 8, y + 24), (x + 32, y - 28)], fill=blanc, width=7)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    img.save(chemin)


def main():
    DATA.mkdir(exist_ok=True)
    (DATA / "manuel_px200.json").write_text(json.dumps(MANUEL, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA / "photos_clients.json").write_text(json.dumps(PHOTOS, ensure_ascii=False, indent=2), encoding="utf-8")

    db = DATA / "registre_garanties.db"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE garanties (serie TEXT PRIMARY KEY, produit TEXT, date_achat TEXT, duree_mois INTEGER)")
    con.executemany("INSERT INTO garanties VALUES (?,?,?,?)", REGISTRE)
    con.commit()
    con.close()

    try:
        orange, rouge, bleu, vert = (255, 140, 0), (220, 40, 40), (40, 120, 255), (40, 190, 80)
        dessiner(DATA / "figures/fig_01.png", orange, "papier", True)
        dessiner(DATA / "figures/fig_02.png", rouge, "goutte", False)
        dessiner(DATA / "figures/fig_03.png", bleu, "ondes", True)
        dessiner(DATA / "figures/fig_04.png", vert, "coche", False)
        dessiner(DATA / "photos/img_px200_01.png", orange, "papier", True)
    except ImportError:
        print("Pillow absent : images non generees (pip install pillow). Le reste fonctionne.")
    print(f"Donnees creees dans {DATA}")


if __name__ == "__main__":
    main()
