from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore")
from langchain.vectorstores import MongoDBAtlasVectorSearch
import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader('text_file/text.txt')
docs = loader.load()

text_spiliter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=2)
splits = text_spiliter.split_documents(docs)

embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=os.getenv('MISTRAL_AI_API_KEY'))

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
    
# for i in store:
#     embed = embeddings.embed_documents([i])
#     embedding.append({"content": i, "embedding": embed})



