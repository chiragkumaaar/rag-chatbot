
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


loader = PyMuPDFLoader("sample.pdf")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", ".", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="./db")

db.persist()

print("✅ Embeddings created and saved to ./db folder.")
