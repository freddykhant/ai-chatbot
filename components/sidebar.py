import streamlit as st

from components.tabs.source import source
from components.tabs.about import about

def sidebar():
  with st.sidebar:
    tab1, tab2 = st.sidebar.tabs(["Data Sources", "About"])

    with tab1:
      source()

    with tab2:
      about()