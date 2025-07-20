import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./db", embedding_function=embedding_model)
retriever = db.as_retriever()

llm = OpenAI(
    temperature=0,
    openai_api_key=openai_key,
    streaming=True,
    verbose=True,
)

st.title("Just a Chatbot")
query = st.text_input("Ask a question about your document:")

if query:
   
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)

  
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}\nAnswer:"

    def generate():
        for chunk in llm.stream(prompt):
            yield chunk  

    st.write_stream(generate)

