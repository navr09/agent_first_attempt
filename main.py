# from sanskritRAG.documentParser import load_and_chunk_sanskrit_pdfs
from sanskritRAG.sanskritModel import load_sanskrit_model
# from sanskritRAG.vectorDB import create_sanskrit_vector_db
from sanskritRAG.RAGchain import setup_rag_chain
from config import HF_TOKEN

from huggingface_hub import whoami

def main():
    # 1. Process Sanskrit PDFs
    # pdf_paths = ["path/to/sanskrit1.pdf", "path/to/sanskrit2.pdf"]  # Add your PDF paths
    # chunks = load_and_chunk_sanskrit_pdfs(pdf_paths)
    
    # 2. Create vector database
    # vector_db = create_sanskrit_vector_db(chunks)
    
    # 3. Load Sanskrit model
    sanskrit_llm = load_sanskrit_model()
    
    # 4. Setup RAG chain
    # qa_chain = setup_rag_chain(vector_db, sanskrit_llm)
    qa_chain = setup_rag_chain(sanskrit_llm)
    
    # 5. Voice interaction loop
    while True:
        # Get voice input
        # query = voice_to_text()
        query = "अहं संस्कृतं पठामि। भवान् किं पठति?"
        if not query:
            continue
            
        # Get RAG response
        result = qa_chain({"query": query})
        answer = result["result"]
        sources = result["source_documents"]
        
        print("Answer:", answer)
        # print("Sources:", [s.metadata["source"] for s in sources])
        
        # Voice output
        # text_to_voice(answer)
        
        if "exit" in query.lower():
            break

if __name__ == "__main__":
    # main()
    # print(whoami(token=HF_TOKEN)) 
    from transformers import pipeline
    # Option 1: Use IndicTrans2 (if you have access)
    # llm = pipeline("text2text-generation", model="ai4bharat/indictrans2-en-indic-ssft")

    # Option 2: Use a multilingual model (less ideal)
    llm = pipeline("text-generation", model="facebook/mbart-large-50",token=HF_TOKEN )

    # Test
    response = llm("भगवद्गीतायाः मुख्यः उपदेशः कः?")
    print(response[0]["generated_text"])    
