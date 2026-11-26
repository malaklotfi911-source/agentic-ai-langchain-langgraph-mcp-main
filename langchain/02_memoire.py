"""A.2 - Ajouter une memoire a l'agent (checkpointer + thread_id)."""
from config import get_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=get_model(),
    tools=[],
    system_prompt="Tu es un assistant utile. Reponds en francais.",
    checkpointer=InMemorySaver(),
)

config1 = {"configurable": {"thread_id": "conversation-1"}}
config2 = {"configurable": {"thread_id": "conversation-2"}}

print("=== Conversation 1 : l'agent se souvient ===")
agent.invoke({"messages": [{"role": "user", "content": "Je m'appelle Mehdi."}]}, config1)
r = agent.invoke({"messages": [{"role": "user", "content": "Comment je m'appelle ?"}]}, config1)
r["messages"][-1].pretty_print()

print("\n=== Conversation 2 : autre thread, memoire separee ===")
r = agent.invoke({"messages": [{"role": "user", "content": "Comment je m'appelle ?"}]}, config2)
r["messages"][-1].pretty_print()

# Memoire persistante (survit au redemarrage) : remplacer InMemorySaver par
# SqliteSaver (pip install langgraph-checkpoint-sqlite).
