from utils.setup import create_llm, create_embeddings
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
  st.caption("Model loaded ✅")

  embedding_model = st.session_state["embedding_model"]
  embeddings = create_embeddings(embedding_model)

  if st.session_state["documents"] is not None and len(st.session_state["documents"]) > 0:
    st.caption("Documents Processed ✅")
  else:
    save_dir = os.getcwd() + "/uploads"
    documents = load_documents(save_dir)
    st.session_state["documents"] = documents
    st.caption("Data Processed ✅")

  create_retriever(st.session_state["documents"], embeddings)