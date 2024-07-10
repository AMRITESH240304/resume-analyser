from pymongo import MongoClient
import timeout_decorator
import argparse
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.document_compressors import LLMChainExtractor
import warnings
from dotenv import load_dotenv
import os

load_dotenv()
warnings.filterwarnings('ignore')

client = MongoClient(os.getenv('MONGODB_URI'))
collection = client['LangChain']['vectors']

parser = argparse.ArgumentParser(description='Atlas Vector Search Demo')
parser.add_argument('-q', '--question', help="The question to ask")
args = parser.parse_args()
query = args.question

print("\nYour question:")
print("-------------")
print(query)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv('GEMINI_API_KEY'))

vectorStore = MongoDBAtlasVectorSearch(collection, embeddings, index_name="vector_index")

print("---------------")

@timeout_decorator.timeout(5, timeout_exception=StopIteration)
def get_docs(query):
    try:
        print("inside the function")
        print(query)
        docs = vectorStore.max_marginal_relevance_search(query, K=1)
        print("Vector Search Results:")
        print(docs[0].page_content)
    except Exception as e:
        print("Database timeout or error:", str(e))

get_docs(query)

llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv('GEMINI_API_KEY'), temperature=0.6)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorStore.as_retriever()
)

print("\nAI Response:")

compressed_docs = compression_retriever.get_relevant_documents(query)
if compressed_docs:
    print(compressed_docs[0].page_content)
else:
    print("No relevant documents found.")

print("\nDebug Information:")
print("Original Query Length:", len(query))
