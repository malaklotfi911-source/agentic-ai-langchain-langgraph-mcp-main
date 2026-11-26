"""Outil LOCAL de l'agent garantie."""
import json

from langchain.tools import tool

from registre import verifier_garantie


@tool
def consulter_garantie(numero_serie: str) -> str:
    """Consulte le registre (lecture seule) et indique si l'appareil est sous garantie a partir de son numero de serie (ex: PX200-0042)."""
    return json.dumps(verifier_garantie(numero_serie), ensure_ascii=False)
