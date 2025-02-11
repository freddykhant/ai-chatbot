import os
from langchain_community.document_loaders import DirectoryLoader
from utils.setup import create_text_splitter
from langchain_chroma import Chroma
import streamlit as st

def load_documents(directory: str):
  documents = []
  loader = DirectoryLoader(directory)
  documents += loader.load()
    
  return documents

def create_vector_store(_documents, _embeddings):
  text_splitter = create_text_splitter()
  chunks = text_splitter.split_documents(_documents)
  
  # Create the Chroma vector store with the embedding function
  vectorstore = Chroma(persist_directory="vectorstore", embedding_function=_embeddings)
  vectorstore.add_documents(documents=chunks)
  
  return vectorstore

def create_retriever(documents, embeddings):
  vectorstore = create_vector_store(documents, embeddings)
  retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 20, "score_threshold": 0.1},
  )

  st.session_state["retriever"] = retriever

  return retriever