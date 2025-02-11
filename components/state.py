import streamlit as st

def set_initial_state():
  ### general ###
  if "messages" not in st.session_state:
    st.session_state["messages"] = [
      {
        "role": "assistant",
        "content": "Hello! I'm the LLama 3.1 Chatbot. How can I help you today?",
      }
    ]

  ### RAG ###
  if "llm" not in st.session_state:
    st.session_state["llm"] = None

  if "documents" not in st.session_state:
    st.session_state["documents"] = []

  if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

  ### files
  if "files" not in st.session_state:
    st.session_state["files"] = []