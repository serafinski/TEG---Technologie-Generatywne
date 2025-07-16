import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from config_manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.get_config()

load_dotenv()

def get_llm(provider, model, temperature, system_message=None):
    """
    Create a language model instance based on the provider
    """
    if provider == "openai":
        # Check if the model doesn't support temperature (like o3-mini)
        if model in ["o3-mini", "o1-mini"]:
            return ChatOpenAI(model=model)
        else:
            return ChatOpenAI(model=model, temperature=temperature)
    elif provider == "anthropic":
        try:
            # Try to create with system message if provided
            if system_message:
                return ChatAnthropic(model=model, temperature=temperature, system=system_message)
            else:
                return ChatAnthropic(model=model, temperature=temperature)
        except Exception:
            # If system parameter fails, create without it
            return ChatAnthropic(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def chain(vector_store: Chroma, provider=None, base_model=None, temperature=None, k=None, system_message=None):
    # Use provided parameters or fall back to config values
    model_provider = provider or config.get('default_provider', 'openai')
    model = base_model or config.get('default_model')
    temp = temperature if temperature is not None else config.get('temperature')
    k_value = k if k is not None else config.get('k')
    
    # Use provided system message or fall back to config
    system_msg = system_message or config.get('default_system_message')
    
    # Get the appropriate LLM
    llm = get_llm(model_provider, model, temp, system_msg)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": k_value})

    # Prepare the prompt template
    prompt_template = system_msg + "\nQuestion: {question}" + "\n\nContext: {context}"

    rag_prompt_template = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"]
    )

    # Create the RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": rag_prompt_template},
        return_source_documents=True
    )

    return rag_chain

def query_rag_chain(rag_chain, query):
    result = rag_chain.invoke(query)

    return{
        "query": query,
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }

def update_chain_parameters(rag_chain, provider=None, model=None, temperature=None, k=None, system_message=None):
    # Get the vector store from the existing chain
    vector_store = rag_chain.retriever.vectorstore
    
    # Create a new chain with updated parameters
    return chain(
        vector_store=vector_store,
        provider=provider,
        base_model=model,
        temperature=temperature,
        k=k,
        system_message=system_message
    )