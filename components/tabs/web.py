import streamlit as st
import utils.RAG as rag
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urlparse

def ensure_https(url):
  parsed = urlparse(url)
  if not bool(parsed.scheme):
    return f"https://{url}"

  return url

def website():
  st.write("Enter a website URL to scrape:")
  col1, col2 = st.columns([1, 0.2])
  with col1:
    url_text = st.text_input("Enter a website", label_visibility="collapsed")
  with col2:
    add_button = st.button("➕")

  if add_button and url_text != "":
    st.session_state["websites"].append(ensure_https(url_text))
    st.session_state["websites"] = sorted(set(st.session_state["websites"]))