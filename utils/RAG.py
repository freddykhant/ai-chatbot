from utils.setup import create_llm, create_embeddings
from utils.retriever import load_documents, create_retriever
import os
import streamlit as st

def rag_pipeline(uploaded_files: list=None):
  if uploaded_files is not None:
    for uploaded_file in uploaded_files:
      save_dir = os.path.join("uploads")
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

  llm = create_llm()
  st.session_state["llm"] = llm

  embeddings = create_embeddings()
  st.session_state["embeddings"] = embeddings

  save_dir = os.getcwd() + "/uploads"
  documents = load_documents(save_dir)
  st.session_state["documents"] = documents

  create_retriever(st.session_state["documents"], st.session_state["embeddings"])