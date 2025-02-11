import streamlit as st

def set_initial_state():
  ### general ###
  if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = "nomic-embed-text-v1.5"

  ### RAG ###
  if "llm" not in st.session_state:
    st.session_state["llm"] = None

  if "documents" not in st.session_state:
    st.session_state["documents"] = []

  if "retriever" not in st.session_state:
    st.session_state["retriever"] = None