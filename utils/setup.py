from langchain_ollama import ChatOllama
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from utils.helper import format_docs

chat_prompt = PromptTemplate.from_template(
  """
  You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
  If you don't know the answer, just say that you don't know. 

  Question: {question}

  Context: {context}
  """
)

# functions for setup
def create_llm():
  return ChatOllama(model="llama3.1:8b", temperature=0)

def create_embeddings():
  return NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

def create_text_splitter():
  return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def chat_context(prompt, retriever, llm):
  docs = retriever.invoke(prompt)
  docs_txt = format_docs(docs)
  chat_prompt_formatted = chat_prompt.format(context=docs_txt, question=prompt)

  return (chunk.content for chunk in llm.stream(chat_prompt_formatted))

def chat(prompt, llm):
  return (chunk.content for chunk in llm.stream(prompt))