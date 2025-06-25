from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

def setup_rag_chain(llm_pipeline, vector_db=None):
    if vector_db:
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline=llm_pipeline),
        chain_type="stuff",
        # retriever=retriever,
        # return_source_documents=True
    )
    
    return qa_chain