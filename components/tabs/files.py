import streamlit as st
import utils.RAG as rag

supported_files = (
    "csv",
    "docx",
    "epub",
    "ipynb",
    "json",
    "md",
    "pdf",
    "ppt",
    "pptx",
    "txt",
)

def local_files():
  uploaded_files = st.file_uploader("Upload a file", type=supported_files, accept_multiple_files=True)
  
  if len(uploaded_files) > 0:
    st.session_state["files"] = uploaded_files

    with st.spinner("Processing"):
      rag.rag_pipeline(uploaded_files)
      st.write("Your files are ready. Let's chat!")