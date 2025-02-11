import streamlit as st

from components.tabs.files import local_files 

def source():
  st.title("Source")
  st.caption("Convert your data into embeddings for utilisation during chat")
  st.write("")

  with st.expander("💻 &nbsp; **Local Files**", expanded=False):
        local_files()

        