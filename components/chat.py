import streamlit as st
from utils.setup import chat

def chatbot():
  if prompt := st.chat_input("How can I help?"):
    if not st.session_state["retriever"]:
      st.error("Please upload a document first")
      st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        response = st.write_stream(
          chat(
            prompt=prompt,
            retriever=st.session_state["retriever"]
          )
        )

    st.session_state["messages"].append({"role": "assistant", "content": response})
  