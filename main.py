import streamlit as st

from components.chat import chatbot
from components.sidebar import sidebar
from components.header import set_page_header
from components.state import set_initial_state
import os 

os.environ["STREAMLIT_WATCH_DISABLE"] = "true"
os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; MyStreamlitApp/1.0)"

set_initial_state()
set_page_header()

for msg in st.session_state["messages"]:
  st.chat_message(msg["role"]).write(msg["content"])

sidebar()

chatbot()