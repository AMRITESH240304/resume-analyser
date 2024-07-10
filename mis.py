from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore")
from langchain.vectorstores import MongoDBAtlasVectorSearch
import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader('text_file/full_text.txt')
docs = loader.load()

text_spiliter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_spiliter.split_documents(docs)
# print(splits[0].page_content)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv('GEMINI_API_KEY'))

# embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=os.getenv('MISTRAL_AI_API_KEY'))

store = [] 
embedding = []

for split in splits:
    store.append(split.page_content)
    
client = MongoClient(os.getenv('MONGODB_URI'))
collection = client['LangChain']['vectors']

collection.delete_many({})

docsearch = MongoDBAtlasVectorSearch.from_documents(
    splits, embeddings, collection=collection, index_name="embeddings"
)
    
print(docsearch)
print("Embeddings saved to MongoDB vector base")




