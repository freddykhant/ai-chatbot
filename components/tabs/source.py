import streamlit as st

from components.tabs.files import local_files 
from components.tabs.web import website

def source():
  st.title("Source")
  st.caption("Convert your data into embeddings for utilisation during chat")
  st.write("")

  with st.expander("💻 &nbsp; **Local Files**", expanded=False):
        local_files()

  with st.expander("🔗 &nbsp; **URLs**", expanded=False):
        website()