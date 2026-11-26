"""Serveur MCP (streamable-http) exposant deux tools.
Lancement : python server.py  ->  http://127.0.0.1:8000/mcp
"""
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

mcp = FastMCP(
    "sav-demo",
    host="0.0.0.0",
    port=8000,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

CATALOGUE = {
    "px-200": {"nom": "Imprimante PX-200", "prix_ht": 1500.0, "stock": 12},
    "px-300": {"nom": "Imprimante PX-300", "prix_ht": 2400.0, "stock": 0},
    "sc-10": {"nom": "Scanner SC-10", "prix_ht": 900.0, "stock": 5},
}


@mcp.tool()
def rechercher_produit(reference: str) -> dict:
    """Retourne le nom, le prix HT et le stock d'un produit à partir de sa référence (ex: PX-200)."""
    produit = CATALOGUE.get(reference.strip().lower())
    if produit is None:
        return {"erreur": f"Référence inconnue : {reference}"}
    return produit


@mcp.tool()
def calculer_prix_ttc(prix_ht: float, taux_tva: float = 20.0) -> float:
    """Calcule le prix TTC à partir du prix HT et du taux de TVA en pourcentage (20 par défaut)."""
    return round(prix_ht * (1 + taux_tva / 100), 2)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")