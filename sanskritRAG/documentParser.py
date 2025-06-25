from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_sanskrit_pdfs(pdf_paths):
    # Load PDFs
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    
    # Sanskrit-specific text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "।", "॥", " ", ""]  # Sanskrit sentence endings
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks