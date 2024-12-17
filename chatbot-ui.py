from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

template = """

You are a helpful assistant, please answer any user queries to the best of your ability.

Question: {question}

Answer:
"""

# App title
st.title('Ask Llama 3.1')

# Session state message to hold old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Get input from user
input_text = st.chat_input("Chat with Llama 3.1")

# Set up our LLM model with Ollama
model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Conversation logic
if input_text:
    # display input prompt
    st.chat_message('user').markdown(input_text)
    # store user input in state
    st.session_state.messages.append({'role':'user', 'content':input_text})
    # invoke LLM chain
    response = chain.invoke({"question": input_text})
    # show LLM response
    st.chat_message('assistant').markdown(response)
    # store LLM response in session state
    st.session_state.messages.append(
        {
            'role':'assistant',
            'content':response
        }
    )