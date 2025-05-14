
import chromadb 
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings

if __name__ == '__main__': 

    # Set up OCI Generative AI LLM 
    llm = OCIGenAI(
        model_id="",
        service_endpoint="",
        compartment_id="",
        model_kwargs={}
    )

    # Set up and connect to ChromaDB server 
    client = chromadb.HttpClient(host="127.0.0.1")

    # Create document embeddings 
    embeddings = OCIGenAIEmbeddings(
        model_id="",
        service_endpoint="",
        compartment_id=""
    )

    chroma_db = Chroma(
        client=client,
        embedding_function=embeddings
    )

    # Set up a retriever to fetch relevant documents
    retriever = chroma_db.as_retriever(search_type="similarity",
                                       search_kwargs={"k": 5})

    # Explore how similar documents are to the query by inspecting metadata
    docs = retriever.get_relevant_documents("PROMPT HERE")
    for doc in docs: 
        print(doc.metadata)
    
    # Create retrieval chain that takes llm retriever and invokes it to get response to the query 
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        retriever=retriever,
                                        return_source_documents=True)
    
    response = chain.invoke("PROMPT HERE (SAME)")
    print(response)