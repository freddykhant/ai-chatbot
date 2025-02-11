import streamlit as st

def about():
  st.title("Llama 3.1 RAG Chatbot 🦙📚")
  st.caption("Built by Freddy Khant")
  st.write("")

  links_html = """
  <ul style="list-style-type: none; padding-left: 0;">
    <li>
      <a href="https://github.com/freddykhant" style="color: grey;">GitHub</a>
    </li>
  </ul>
  """

  st.subheader("Links")
  st.markdown(links_html, unsafe_allow_html=True)