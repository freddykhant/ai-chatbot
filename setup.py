from langchain_ollama import ChatOllama
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# llm setup
model = "llama3.1:8b"
llm = ChatOllama(model=model, temperature=0)

# embeddings setup
emb_model = "nomic-embed-text-v1.5"
embeddings = NomicEmbeddings(model=emb_model, inference_mode="local")

# text splitter setup
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000, chunk_overlap=200
)

chat_prompt = PromptTemplate.from_template(
  """
  You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
  If you don't know the answer, just say that you don't know. 

  Question: {question}

  Context: {context}
  """
)