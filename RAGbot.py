from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import os
import getpass
import json
from dotenv import load_dotenv

load_dotenv()

# LLM setup
local_llm = "llama3.2:3b"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

# def _set_env(var: str):
#   if not os.environ.get(var):
#     os.environ[var] = getpass.getpass(f"{var}: ")
  
# _set_env("TAVILY_API_KEY")
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

# # Set User Agent
# os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"

# Set headers for User Agent
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"}

urls = [
  "https://www.travelandleisure.com/food-drink/restaurants/best-ramen-chain-restaurants-goo-ranking?",
  "https://www.tasteatlas.com/ramen/wheretoeat?",
  "https://www.petitegourmets.com/food-blog/best-ramen-noodle-restaurants-in-the-world?"
]

# Load documents
docs = []
for url in urls:
    docs.extend(WebBaseLoader(url).load())

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
   chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs)

# Add to vectorDB
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits, 
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
)

# Create retriever
k = min(3, len(doc_splits))  # Ensure k does not exceed available chunks
retriever = vectorstore.as_retriever(k=k)

# Retrieve
result = retriever.invoke("best ramen restaurants")
print(result)