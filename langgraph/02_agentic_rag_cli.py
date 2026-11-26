"""B.2 - Test en console du chatbot Agentic RAG (avant LangGraph Studio et Streamlit).

Prerequis : copier un PDF dans le dossier documents/  puis  python 02_agentic_rag_cli.py
"""
from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag.graph import builder

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "cli-1"}}

print("Chatbot Agentic RAG (tape 'quit' pour sortir)")
while True:
    question = input("\nVous : ")
    if question.strip().lower() in {"quit", "exit", "q"}:
        break
    reponse = None
    for update in graph.stream({"messages": [("user", question)]}, config, stream_mode="updates"):
        for node, contenu in update.items():
            print(f"  -> etape : {node}")
            if contenu and contenu.get("messages"):
                m = contenu["messages"][-1]
                if node in ("generate_answer", "generate_query_or_respond") and not getattr(m, "tool_calls", None):
                    reponse = m.content
    print(f"\nAssistant : {reponse}")
