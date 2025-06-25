from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_sanskrit_vector_db(chunks):
    # Sanskrit-optimized embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Create and save vector store
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("sanskrit_vector_db")
    return vector_db