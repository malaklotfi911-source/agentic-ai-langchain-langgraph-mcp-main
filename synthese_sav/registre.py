"""Acces en LECTURE SEULE au registre des garanties.

Test rapide : python registre.py PX200-0042
"""
import calendar
import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

DB = Path(__file__).resolve().parent / "data" / "registre_garanties.db"


def _ajouter_mois(d: date, mois: int) -> date:
    y = d.year + (d.month - 1 + mois) // 12
    m = (d.month - 1 + mois) % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def verifier_garantie(numero_serie: str, aujourdhui: date | None = None) -> dict:
    serie = numero_serie.strip().upper()
    con = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)   # ouverture en lecture seule
    try:
        con.execute("PRAGMA query_only = ON")
        row = con.execute(
            "SELECT serie, produit, date_achat, duree_mois FROM garanties WHERE serie = ?", (serie,)
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return {"serie": serie, "trouve": False, "message": "Numero de serie inconnu dans le registre."}

    _, produit, date_achat, duree = row
    fin = _ajouter_mois(datetime.strptime(date_achat, "%Y-%m-%d").date(), duree)
    jours = (fin - (aujourdhui or date.today())).days
    return {
        "serie": serie, "trouve": True, "produit": produit, "date_achat": date_achat,
        "duree_mois": duree, "date_fin_garantie": fin.isoformat(),
        "sous_garantie": jours >= 0,
        "jours_restants" if jours >= 0 else "jours_depuis_expiration": abs(jours),
    }


if __name__ == "__main__":
    print(json.dumps(verifier_garantie(sys.argv[1] if len(sys.argv) > 1 else "PX200-0042"), ensure_ascii=False, indent=2))
