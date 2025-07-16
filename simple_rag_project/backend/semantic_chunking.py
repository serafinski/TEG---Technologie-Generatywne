from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from backend.pdf_loader import get_documents_from_directory
from backend.chain import chain
from backend.vector_store import create_vector_db

class SemanticChunkingProcessor:
    """
    Implements semantic chunking for document processing using LangChain's SemanticChunker.
    Divides text into more meaningful and context-aware segments rather than fixed character counts.
    """
    def __init__(self, 
                 breakpoint_type='percentile', 
                 threshold_amount=90, 
                 k=2,
                 provider=None,
                 model=None,
                 temperature=None):
        self.embeddings = OpenAIEmbeddings()
        self.breakpoint_type = breakpoint_type
        self.threshold_amount = threshold_amount
        self.k = k
        self.provider = provider
        self.model = model
        self.temperature = temperature
        
        # Initialize the semantic chunker
        self.text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=threshold_amount
        )
        
    def process_directory(self, directory_path):
        """Process all documents in a directory using semantic chunking"""
        try:
            # Get all documents from the directory
            all_docs = get_documents_from_directory(directory_path, verbose=True)
            
            if not all_docs:
                print(f"No documents found in {directory_path}")
                return None
            
            # Extract all text content from documents
            all_text = []
            for doc in all_docs:
                all_text.append(doc.page_content)
            
            # Create semantic chunks
            semantic_chunks = self.text_splitter.create_documents(all_text)
            print(f"Created {len(semantic_chunks)} semantic chunks")
            
            # Create vector store using ChromaDB
            vector_db = create_vector_db(semantic_chunks, collection_name="chroma")
            
            # Create chain
            rag_chain = chain(
                vector_db, 
                provider=self.provider,
                base_model=self.model,
                temperature=self.temperature,
                k=self.k
            )
            
            return {
                "directory_path": directory_path,
                "semantic_chunks": semantic_chunks,
                "vector_db": vector_db,
                "rag_chain": rag_chain
            }
            
        except Exception as e:
            print(f"Error in semantic chunking: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def retrieve(self, query):
        """Retrieve relevant chunks for a query"""
        try:
            results = self.retriever.get_relevant_documents(query)
            return {
                "chunks": results,
                "text": [doc.page_content for doc in results]
            }
        except Exception as e:
            print(f"Error retrieving with semantic chunks: {e}")
            return None

def get_semantic_chunking_processor(
    directory_path,
    breakpoint_type='percentile',
    threshold_amount=90,
    k=2,
    provider=None,
    model=None,
    temperature=None
):
    """Create and return a semantic chunking processor instance with processed documents"""
    processor = SemanticChunkingProcessor(
        breakpoint_type=breakpoint_type,
        threshold_amount=threshold_amount,
        k=k,
        provider=provider,
        model=model,
        temperature=temperature
    )
    
    # Process the directory
    result = processor.process_directory(directory_path)
    
    return result