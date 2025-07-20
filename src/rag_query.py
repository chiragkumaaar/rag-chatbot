
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from zephyr_llm import query_zephyr
from langchain.chains import RetrievalQA

openai_key = os.getenv("OPENAI_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./db", embedding_function=embedding_model)

retriever = db.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
query = input("Ask a question about your document: ")
context = retriever.get_relevant_documents(query)
combined_context = "\n".join([doc.page_content for doc in context])
prompt = f"Answer the following question based on the context below:\n\nContext:\n{combined_context}\n\nQuestion: {query}"
result = query_zephyr(prompt)
print("\n💡 Answer:\n", result)
print(result)
