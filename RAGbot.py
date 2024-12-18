from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import os
import getpass
import json
# from dotenv import load_dotenv

# load_dotenv()

# LLM setup
local_llm = "llama3.2:3b"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

def _set_env(var: str):
  if not os.environ.get(var):
    os.environ[var] = getpass.getpass(f"{var}: ")
  
_set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# # Set User Agent
# os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"

# Set headers for User Agent
# headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"}

### Vectorstore

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
# result = retriever.invoke("best ramen restaurants")
# print(result)

### Router 

router_instructions = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to the best ramen restaurants.
Use the vectorstore for questions on these topics. For all else, and especially current events, use web-search.
Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question.
"""

question = [HumanMessage(content="List 3 of the best ramen restaurants?")]
test_vector_store = llm_json_mode.invoke([SystemMessage(content=router_instructions)] + question)
json.loads(test_vector_store.content)

### Retrieval Grader

# Doc grader instructions
doc_grader_instructions = """ You are a grader assessing the relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """ Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.
This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

# Test
question = "What is the best ramen restaurant in the world?"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
doc_grader_prompt_formatted = doc_grader_prompt.format(
   document=doc_txt, question=question
)

result = llm_json_mode.invoke(
    [SystemMessage(content=doc_grader_instructions)]
    + [HumanMessage(content=doc_grader_prompt_formatted)]
)
json.loads(result.content)
