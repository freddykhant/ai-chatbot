from langchain_ollama import ChatOllama
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# variables
# model = "llama3.1:8b"

# emb_model = "nomic-embed-text-v1.5"

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