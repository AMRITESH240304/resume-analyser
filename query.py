from pymongo import MongoClient
import timeout_decorator
# import argparse
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
import warnings
from dotenv import load_dotenv
import os

load_dotenv()
warnings.filterwarnings('ignore')

client = MongoClient(os.getenv('MONGODB_URI'))
collection = client['LangChain']['vectors']

# parser = argparse.ArgumentParser(description='Atlas Vector Search Demo')
# parser.add_argument('-q', '--question', help="The question to ask")
# args = parser.parse_args()
# query = args.question

# print("\nYour question:")
# print("-------------")
# print(query)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv('GEMINI_API_KEY'))

vectorStore = MongoDBAtlasVectorSearch(collection, embeddings, index_name="vector_index")

# print("---------------")


# llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv('GEMINI_API_KEY'), temperature=0.6)

# def get_conversational_chain():
#     prompt_template = """
#             Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
    
#     model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv('GEMINI_API_KEY'), temperature=0.6)
#     prompt = PromptTemplate(template=prompt_template,input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain






# # # compressor = LLMChainExtractor.from_llm(llm)

# # compression_retriever = ContextualCompressionRetriever(
# #     base_compressor=compressor,
# #     base_retriever=vectorStore.as_retriever(search_kwargs={'k': 4})
# # )
# # print(compression_retriever)

# # print("\nAI Response:")

# # compressed_docs = compression_retriever.get_relevant_documents(query)
# # if compressed_docs:
# #     print(compressed_docs[0].page_content)
# # else:
# #     print("No relevant documents found.")

# # print("\nDebug Information:")
# # print("Original Query Length:", len(query))
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import argparse
import os
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
# from langchain.vectorstores import MongoDBAtlasVectorSearch
from dotenv import load_dotenv

@timeout_decorator.timeout(5, timeout_exception=StopIteration)
def get_docs(query):
    try:
        print("inside the function")
        print(query)
        docs = vectorStore.similarity_search(query, K=4)
        print("Vector Search Results:")
        print(len(docs))
        return docs
    except Exception as e:
        print("Database timeout or error:", str(e))


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv('GEMINI_API_KEY'), temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # new_db = FAISS.load_local("faiss_index", embeddings)
    docs = get_docs(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print("Reply: ", response["output_text"])

def main():
    parser = argparse.ArgumentParser(description='Atlas Vector Search Demo')
    parser.add_argument('-q', '--question', help="The question to ask")
    # parser.add_argument('-p', '--pdfs', nargs='+', help="The PDF files to process", required=True)
    args = parser.parse_args()
    
    
    if args.question:
        user_input(args.question)


if __name__ == "__main__":
    main()