import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = PyPDFLoader("sample.pdf") 
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(chunks, embedding_model, persist_directory="./db")
db.persist()
# for i, chunk in enumerate(chunks, start=1):
#     print(f"\n--- Chunk {i} ---")
#     print(f"Word Count: {len(chunk.page_content.split())}")
#     print(chunk.page_content)


# print(f"✅ Successfully split into {len(chunks)} chunks and saved to vector DB.")
