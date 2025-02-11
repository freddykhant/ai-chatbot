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

  if st.session_state["websites"] != []:
    st.markdown(f"<p>Website(s)</p>", unsafe_allow_html=True)
    for site in st.session_state["websites"]:
      st.caption(f"- {site}")
    st.write("")

  process_button = st.button("Process", key="process")

  if process_button:
    loader = WebBaseLoader(st.session_state["websites"])
    docs = loader.load()

    if len(docs) > 0:
      st.session_state["documents"] = docs

      with st.spinner("Processing"):
        rag.rag_pipeline()
        st.write("Your website is ready. Let's chat!")