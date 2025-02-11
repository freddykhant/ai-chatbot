import streamlit as st
import utils.RAG as rag
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urlparse

def ensure_https(url):
  parsed = urlparse(url)
  if not bool(parsed.scheme):
    return f"https://{url}"

  return url

