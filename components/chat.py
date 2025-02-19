import streamlit as st
from utils.setup import chat_context, chat, create_llm

def chatbot():
  if prompt := st.chat_input("How can I help?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    if not st.session_state["retriever"]:
      llm = create_llm()
      st.session_state["llm"] = llm
      with st.chat_message("assistant"):
        response = st.write_stream(chat(prompt, st.session_state["llm"]))
    else:
      with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
          response = st.write_stream(
              chat_context(
                  prompt=prompt,
                  retriever=st.session_state["retriever"],
                  llm=st.session_state["llm"],
              )
          )


    st.session_state["messages"].append({"role": "assistant", "content": response})
  