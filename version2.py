from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

template = """

You are a helpful assistant, please answer any user queries to the best of your ability.

Question: {question}

Answer:
"""

st.title('Llama 3.1 Chatbot')
input_text = st.text_input("Chat with Llama 3.1")

model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

if input_text:
    st.write(chain.invoke({"question": input_text}))