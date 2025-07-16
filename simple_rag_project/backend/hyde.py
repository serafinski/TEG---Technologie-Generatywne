from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from backend.vector_store import get_vector_db
from langchain_core.messages import HumanMessage

class HyDERetriever:
    """
    HyDE (Hypothetical Document Embeddings) Retriever
    Creates hypothetical documents as if they were perfect answers to the query,
    then uses these to search the vector store for similar actual documents.
    """
    def __init__(self, provider, model, temperature=0, chunk_size=500):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.vector_db = get_vector_db(collection_name="chroma")
        
        # Initialize LLM based on provider
        if provider == "openai":
            # Check if the model doesn't support temperature (like o3-mini, o1-mini)
            known_no_temp_models = ["o3-mini", "o1-mini"]
            
            if model in known_no_temp_models:
                self.llm = ChatOpenAI(model=model, max_tokens=4000)
            else:
                try:
                    self.llm = ChatOpenAI(temperature=temperature, model=model, max_tokens=4000)
                except Exception as e:
                    if "temperature" in str(e).lower():
                        # If we get a temperature-related error, try without temperature
                        print(f"Model {model} doesn't support temperature, creating without it")
                        self.llm = ChatOpenAI(model=model, max_tokens=4000)
                    else:
                        raise e
        else:
            try:
                self.llm = ChatAnthropic(temperature=temperature, model=model)
            except Exception as e:
                if "temperature" in str(e).lower():
                    # If we get a temperature-related error, try without temperature
                    print(f"Model {model} doesn't support temperature, creating without it")
                    self.llm = ChatAnthropic(model=model)
                else:
                    raise e
        
        # Create the prompt template string for generating hypothetical documents
        self.hyde_template = """Given the question '{query}', generate a hypothetical document that directly answers this question. 
        The document should be detailed and in-depth.
        The document size should be approximately {chunk_size} characters."""

    def generate_hypothetical_document(self, query):
        """Generate a hypothetical document that answers the query"""
        try:
            # For models with message role restrictions, use a direct approach with just user role messages
            if self.provider == "openai" and self.model in ["o3-mini", "o1-mini"]:
                # Use only a user message to avoid role issues
                messages = [
                    HumanMessage(content=self.hyde_template.format(query=query, chunk_size=self.chunk_size))
                ]
                
                # Invoke the LLM directly
                result = self.llm.invoke(messages)
                return result.content
            else:
                # Use the standard approach with a prompt template for other models
                prompt_template = PromptTemplate(
                    input_variables=["query", "chunk_size"],
                    template=self.hyde_template
                )
                
                # Create the chain for generating hypothetical documents
                hyde_chain = prompt_template | self.llm
                
                # Generate the document
                input_variables = {"query": query, "chunk_size": self.chunk_size}
                return hyde_chain.invoke(input_variables).content
                
        except Exception as e:
            # Fall back to simplest possible approach if we encounter errors
            print(f"Error in generate_hypothetical_document: {e}")
            print("Trying fallback approach...")
            
            # Use the most basic, compatible approach
            simple_prompt = f"Write a detailed document answering this question: {query}"
            result = self.llm.invoke([HumanMessage(content=simple_prompt)])
            return result.content

    def retrieve(self, query, k=3):
        """Retrieve documents similar to a hypothetical perfect answer"""
        try:
            # Generate hypothetical document
            hypothetical_doc = self.generate_hypothetical_document(query)
            
            # Search for similar documents
            similar_docs = self.vector_db.similarity_search(hypothetical_doc, k=k)
            
            return {
                "documents": similar_docs,
                "hypothetical_doc": hypothetical_doc
            }
        except Exception as e:
            print(f"Error in HyDE retrieval: {e}")
            # Fallback to regular search if HyDE fails
            similar_docs = self.vector_db.similarity_search(query, k=k)
            return {
                "documents": similar_docs,
                "hypothetical_doc": f"Error generating hypothetical document: {str(e)}\nFalling back to direct query retrieval."
            }

def get_hyde_retriever(provider, model, temperature=0, chunk_size=500):
    """Create and return a HyDE retriever instance"""
    return HyDERetriever(
        provider=provider,
        model=model,
        temperature=temperature,
        chunk_size=chunk_size
    )