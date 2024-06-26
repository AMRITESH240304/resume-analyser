from pymongo import MongoClient
import argparse
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_mistralai import MistralAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_mistralai import ChatMistralAI
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

embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=os.getenv('MISTRAL_AI_API_KEY'))

vectorStore = MongoDBAtlasVectorSearch(collection, embeddings, index_name="vector_index")

print("---------------")

docs = vectorStore.max_marginal_relevance_search(query, K=1)
print("Vector Search Results:")
print(docs[0].page_content)

llm = ChatMistralAI(mistral_api_key=os.getenv('MISTRAL_AI_API_KEY'), temperature=0.6)
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
