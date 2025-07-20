This is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain and Streamlit. It allows users to query the content of uploaded documents using natural language. The pipeline uses sentence-aware chunking, HuggingFace embeddings, Chroma for vector storage, and OpenAI's GPT for response generation. Responses are streamed in real-time via Streamlit's write_stream().

Project Architecture and Flow

📁chatbot/
├── data/              # Raw input documents (e.g., PDFs)
├── chunks/            # Preprocessed, chunked text segments
├── vectordb/          # Chroma vector database with embeddings
├── src/
│   ├── chunk_text.py         # Loads + splits text
│   ├── embed_store.py        # Creates and saves embeddings
│   └── rag_query.py          # CLI chatbot (non-streaming)
├── app.py             # Streamlit chatbot with streaming
├── requirements.txt   # All dependencies
└── README.md          # You're here!

Steps to Run
1.  Preprocessing (Chunking)
    python src/chunk_text.py
    This splits the documents into 100–300 word segments called chunks.
2.  Create Embeddings + Store in Chroma
    python src/embed_store.py
3. Run RAG Chatbot via CLI
    python src/rag_query.py
4.  Run RAG Chatbot via Streamlit
    streamlit run app.py
    Make sure to set your OPENAI_API_KEY in a .env file.

   Model & Embedding Choices
Component	           Choice	                                         Reason
Embeddings	     all-MiniLM-L6-v2 (HuggingFace)	       Lightweight, sentence-aware, fast
LLM	             gpt-3.5-turbo via langchain_openai	           Reliable and accurate text generation
Vector DB	     Chroma	                               Simple, local, high-perf vector storage
Splitter	     RecursiveCharacterTextSplitter	       Maintains semantic coherence in chunks


How to Use Streaming Chatbot
 streamlit run app.py


Demo Video Link
https://youtu.be/69gwti2P1oc





