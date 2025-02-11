import os
from langchain_community.document_loaders import DirectoryLoader
from setup import creatllm, create_embeddings, create_text_splitter
from langchain_chroma import Chroma
import streamlit as st

def load_documents(directory: str):
  documents = []
  for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    loader = DirectoryLoader(file_path)
    documents += loader.load()
    
  return documents

def create_vector_store(documents):
  text_splitter = create_text_splitter()
  chunks = text_splitter.split(documents)
  vectorstore = Chroma.from_documents(
    documents=chunks,
    embeddings=create_embeddings(),
    persist_directory="vectorstore"
  )
  
  return vectorstore

def create_retriever(documents):
  vectorstore = create_vector_store(documents)
  retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 20, "score_threshold": 0.1},
  )

  st.session_state["retriever"] = retriever

  return retriever  