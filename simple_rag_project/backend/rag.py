import os
from backend.pdf_loader import get_documents_from_directory
from backend.splitter import split_documents
from backend.vector_store import create_vector_db
from backend.chain import chain, update_chain_parameters

def rag(directory: str, verbose: bool = False, provider=None, model=None, temperature=None, k=None, system_message=None, rebuild=False):
    
    # Path to check if the vector store exists
    vector_store_path = os.path.join("db/", "chroma")
    
    # If rebuild or vector store doesn't exist, create it
    if rebuild or not os.path.exists(vector_store_path) or not os.path.isdir(vector_store_path) or len(os.listdir(vector_store_path)) == 0:
        # Create new vector store
        all_docs = get_documents_from_directory(directory, verbose=verbose)
        chunked_docs = split_documents(all_docs, verbose=verbose)
        db = create_vector_db(chunked_docs, collection_name="chroma")
    else:
        # Load existing vector store
        from backend.vector_store import get_vector_db
        db = get_vector_db(collection_name="chroma")
        chunked_docs = None  # Not available when loading
    
    # Create chain
    rag_chain = chain(
        db, 
        provider=provider, 
        base_model=model, 
        temperature=temperature, 
        k=k,
        system_message=system_message
    )
    
    return {
        "directory_path": directory,
        "chunked_docs": chunked_docs,
        "vector_db": db,
        "rag_chain": rag_chain
    }

def update_rag_parameters(rag_system, provider=None, model=None, temperature=None, k=None, system_message=None):
    # Update the chain with new parameters
    updated_chain = update_chain_parameters(
        rag_system["rag_chain"],
        provider=provider,
        model=model,
        temperature=temperature,
        k=k,
        system_message=system_message
    )
    
    # Create a new system dictionary with the updated chain
    updated_system = rag_system.copy()
    updated_system["rag_chain"] = updated_chain
    
    return updated_system