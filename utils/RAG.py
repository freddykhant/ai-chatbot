from utils.setup import create_llm
from utils.retriever import load_documents, create_retriever
from utils.helper import save_uploaded_files
import os
import streamlit as st

def rag_pipeline(uploaded_files: list=None):
  if uploaded_files is not None:
    for uploaded_file in uploaded_files:
      with st.spinner(f"Processing {uploaded_file.name}..."):
        save_dir = os.getcwd() + "/uploads"
        save_uploaded_files(uploaded_file, save_dir)

    st.caption("Files Uploaded ✅")

  llm = create_llm()
  st.session_state["llm"] = llm

  embeddings = st.session_state["embedding_model"]

  save_dir = os.getcwd() + "/uploads"
  documents = load_documents(save_dir)
  st.session_state["documents"] = documents

  create_retriever(st.session_state["documents"], embeddings)