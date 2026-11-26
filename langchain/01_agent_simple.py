"""A.1 - Creer un simple agent avec LangChain."""
from config import get_model
from langchain.agents import create_agent

agent = create_agent(
    model=get_model(),
    tools=[],
    system_prompt="Tu es un assistant utile. Reponds toujours en francais, de facon concise.",
)

print("=== Question simple ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explique en deux phrases ce qu'est un agent IA."}]}
)
result["messages"][-1].pretty_print()

print("\n=== Sans memoire : l'agent oublie tout entre deux appels ===")
agent.invoke({"messages": [{"role": "user", "content": "Je m'appelle Mehdi."}]})
r = agent.invoke({"messages": [{"role": "user", "content": "Comment je m'appelle ?"}]})
r["messages"][-1].pretty_print()
