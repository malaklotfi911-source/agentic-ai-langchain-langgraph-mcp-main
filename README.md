# Agentic AI avec LangChain, LangGraph et MCP

Agents IA avec LangChain, LangGraph et MCP (Model Context Protocol) - TP EMSI 2026/2027

## Structure du dépôt

| Dossier / Fichier | Description |
|---|---|
| `langchain/` | Agents et chaînes LangChain : modèles, prompts, outils (tools) |
| `langgraph/` | Agents sous forme de graphes d'états avec LangGraph : noeuds, transitions, mémoire |
| `mcp/` | Serveur et client MCP (Model Context Protocol) pour exposer des outils à un agent |
| `synthese_sav/` | Application de synthèse pour le service après-vente (SAV) |
| `requirements.txt` | Dépendances Python du projet |
| `.gitignore` | Fichiers exclus du dépôt (clés d'API, environnement virtuel, cache) |

## Concepts abordés

- Agents IA et appel d'outils (tool calling)
- Chaînes et prompts avec LangChain
- Orchestration d'agents avec LangGraph (graphes d'états)
- Protocole MCP : serveurs d'outils et clients
- Cas pratique : synthèse de demandes SAV

## Prérequis

- Python 3.10 ou plus
- Une clé d'API pour le modèle de langage utilisé (à renseigner dans un fichier `.env`)

## Installation

```bash
git clone https://github.com/malaklotfi911-source/agentic-ai-langchain-langgraph-mcp-main.git
cd agentic-ai-langchain-langgraph-mcp-main
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Configuration

Créer un fichier `.env` à la racine du projet :

```env
API_KEY=votre_cle_ici
```

Ce fichier est ignoré par Git et ne doit jamais être publié.

## Utilisation

Chaque dossier contient ses propres exemples. Exemple de lancement :
