import streamlit as st

def set_initial_state():
  if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = "nomic-embed-text-v1.5"