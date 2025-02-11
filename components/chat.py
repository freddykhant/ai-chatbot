import streamlit as st

def chatbot():
  if prompt := st.chat_input("How can I help?"):
    if not st.session_state["retriever"]:
      st.error("Please upload a document first")
      st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    