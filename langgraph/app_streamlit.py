"""B.3 - Interface web Streamlit pour le chatbot Agentic RAG.

Lancement (depuis le dossier B_langgraph) :  streamlit run app_streamlit.py
"""
import uuid

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag.graph import builder

st.set_page_config(page_title="Agentic RAG - LangGraph", page_icon="🤖")
st.title("🤖 Chatbot Agentic RAG")
st.caption("LangGraph + Ollama : l'agent decide de chercher dans le document, evalue les passages et reformule si besoin.")


@st.cache_resource
def get_graph():
    return builder.compile(checkpointer=InMemorySaver())


graph = get_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

with st.sidebar:
    st.header("Session")
    if st.button("Nouvelle conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.caption(f"thread : {st.session_state.thread_id[:8]}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Pose ta question sur le document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        etapes, reponse = [], None
        with st.status("Reflexion de l'agent...", expanded=True) as status:
            try:
                for update in graph.stream({"messages": [("user", question)]}, config, stream_mode="updates"):
                    for node, contenu in update.items():
                        etapes.append(node)
                        st.write(f"Etape : `{node}`")
                        if contenu and contenu.get("messages"):
                            m = contenu["messages"][-1]
                            if node in ("generate_answer", "generate_query_or_respond") and not getattr(m, "tool_calls", None):
                                reponse = str(m.content)
                status.update(label="Termine", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Erreur", state="error")
                reponse = f"Erreur : {e}"
        st.markdown(reponse or "(pas de reponse)")
    st.session_state.messages.append({"role": "assistant", "content": reponse or ""})
