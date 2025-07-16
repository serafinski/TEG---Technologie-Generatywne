import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from config_manager import ConfigManager
from backend.chain import chain, query_rag_chain
from backend.rag import update_rag_parameters
from backend.vector_store import get_vector_db, create_vector_db
from backend.pdf_loader import get_documents_from_directory
from backend.splitter import split_documents
from backend.hyde import get_hyde_retriever

config_manager = ConfigManager()
load_dotenv()

class ChatbotApp:
    def __init__(self):
        # Use a single ConfigManager instance
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        
        st.set_page_config(
            page_title=self.config.get('app').get('title'),
            layout=self.config.get('app').get('layout'),
        )

        # Store default values for easy access
        self.default_provider = self.config.get('default_provider', 'openai')
        self.default_model = self.config.get('default_model')
        self.default_system_message = self.config.get('default_system_message')
        self.default_temperature = float(self.config.get('temperature'))
        self.default_k = int(self.config.get('k'))

        # Initialize the LLM based on default provider
        if self.default_provider == 'openai':
            self.llm = ChatOpenAI(model=self.default_model, temperature=self.default_temperature)
        else:
            self.llm = ChatAnthropic(model=self.default_model, temperature=self.default_temperature)

        with st.spinner("Loading or creating the vector database..."):
            self.rag_system = self.initialize_vector_store("backend/docs")

        self._init_session_state()

    def initialize_vector_store(self, directory: str, rebuild=False):
        """
        Check if vector store exists and create it if it doesn't.
        If rebuild is True, always create a new vector store.
        """
        # Path to check if the vector store exists
        vector_store_path = os.path.join("db/", "chroma")
        
        if not rebuild and os.path.exists(vector_store_path) and os.path.isdir(vector_store_path) and len(os.listdir(vector_store_path)) > 0:
            # Vector store exists, just load it
            try:
                st.info("Vector database exists, and was loaded.")
                db = get_vector_db(collection_name="chroma")
                
                # Create chain with the loaded db
                rag_chain = chain(
                    db, 
                    provider=self.default_provider, 
                    base_model=self.default_model, 
                    temperature=self.default_temperature,
                    k=self.default_k,
                    system_message=self.default_system_message
                )
                
                return {
                    "directory_path": directory,
                    "vector_db": db,
                    "rag_chain": rag_chain
                }
            except Exception as e:
                st.warning(f"Error loading existing vector store: {str(e)}. Creating a new one...")
                # If loading fails, proceed to create a new one
        
        # Vector store doesn't exist, loading failed, or rebuild is True
        if rebuild:
            st.info("Rebuilding vector database from documents...")
        else:
            st.info("Creating new vector database from documents...")
            
        try:
            # Process documents
            all_docs = get_documents_from_directory(directory, verbose=True)
            
            if not all_docs:
                st.warning(f"No documents found in {directory}. Vector store will be empty.")
                all_docs = []
            
            # Split documents
            chunked_docs = split_documents(all_docs, verbose=True)
            
            # Create vector DB using the existing function
            db = create_vector_db(chunked_docs, collection_name="chroma")
            
            # Create chain
            rag_chain = chain(
                db, 
                provider=self.default_provider, 
                base_model=self.default_model, 
                temperature=self.default_temperature,
                k=self.default_k,
                system_message=self.default_system_message
            )
            
            return {
                "directory_path": directory,
                "chunked_docs": chunked_docs,
                "vector_db": db,
                "rag_chain": rag_chain
            }
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            import traceback
            st.exception(traceback.format_exc())
            raise e

    def _init_session_state(self):
        try:
            # Initialize provider selection
            if "provider" not in st.session_state:
                st.session_state.provider = self.default_provider
                
            # Initialize model selection
            if "model" not in st.session_state:
                st.session_state.model = self.default_model

            # Initialize message history with system message
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "system", "content": self.default_system_message}
                ]
            # If messages exist but we want to ensure the system message is up to date with config
            elif len(st.session_state.messages) > 0 and st.session_state.messages[0]["role"] == "system":
                # Only update if it hasn't been manually changed by the user
                if "default_system_message" not in st.session_state.custom_config:
                    st.session_state.messages[0]["content"] = self.default_system_message

            # Initialize RAG toggle
            if "use_rag" not in st.session_state:
                st.session_state.use_rag = True
                
            # Initialize chunking method
            if "chunking_method" not in st.session_state:
                st.session_state.chunking_method = self.config.get('chunking', {}).get('default_method', 'standard')
                
            # Initialize HyDE toggle and settings
            if "use_hyde" not in st.session_state:
                st.session_state.use_hyde = st.session_state.chunking_method == 'hyde'
                
            # Initialize HyDE chunk size
            if "hyde_chunk_size" not in st.session_state:
                st.session_state.hyde_chunk_size = 500
                
            # Initialize semantic chunking settings
            if "semantic_breakpoint_type" not in st.session_state:
                st.session_state.semantic_breakpoint_type = self.config.get('chunking', {}).get('semantic', {}).get('breakpoint_type', 'percentile')
                
            if "semantic_threshold_amount" not in st.session_state:
                st.session_state.semantic_threshold_amount = self.config.get('chunking', {}).get('semantic', {}).get('threshold_amount', 90)
                
            # Initialize temperature (ensure it's a float)
            if "temperature" not in st.session_state:
                st.session_state.temperature = self.default_temperature
                
            # Initialize k value (ensure it's an integer)
            if "k" not in st.session_state:
                st.session_state.k = self.default_k
                
            # Initialize custom config tracking
            if "custom_config" not in st.session_state:
                st.session_state.custom_config = {}
                
        except Exception as e:
            st.error(f"Error initializing session state: {str(e)}")
            import traceback
            st.exception(traceback.format_exc())
    
    def display_sidebar(self):
        with st.sidebar:
            st.title("Settings")

            # Provider selection
            available_providers = self.config.get('available_providers', ['openai', 'anthropic'])
            provider = st.selectbox(
                "Select Provider",
                available_providers,
                index=available_providers.index(
                    st.session_state.provider) if st.session_state.provider in available_providers else 0
            )

            # Update provider in session state when changed
            if provider != st.session_state.provider:
                st.session_state.provider = provider
                st.session_state.custom_config['default_provider'] = provider
                self.config_manager.update_runtime_config('default_provider', provider)
                
                # When provider changes, we need to update the model list and default model
                models_for_provider = self.config.get('available_models', {}).get(provider, [])
                if models_for_provider:
                    st.session_state.model = models_for_provider[0]
                    st.session_state.custom_config['default_model'] = models_for_provider[0]
                    self.config_manager.update_runtime_config('default_model', models_for_provider[0])

            # Model selection - filtered by the selected provider
            models_for_provider = self.config.get('available_models', {}).get(provider, [])
            if not models_for_provider:
                st.error(f"No models configured for provider {provider}")
                models_for_provider = ["No models available"]
                
            model = st.selectbox(
                "Select Model",
                models_for_provider,
                index=models_for_provider.index(
                    st.session_state.model) if st.session_state.model in models_for_provider else 0
            )

            # Update model in session state when changed
            if model != st.session_state.model and model in models_for_provider:
                st.session_state.model = model
                st.session_state.custom_config['default_model'] = model
                self.config_manager.update_runtime_config('default_model', model)

            # System message configuration
            system_message = st.text_area(
                "System message",
                value=st.session_state.messages[0]["content"],
                height=150
            )

            # Update system message when changed
            if system_message != st.session_state.messages[0]["content"]:
                st.session_state.messages[0]["content"] = system_message
                st.session_state.custom_config['default_system_message'] = system_message
                self.config_manager.update_runtime_config('default_system_message', system_message)

            # RAG toggle
            use_rag = st.checkbox("Use RAG for answers", value=st.session_state.use_rag)
            if use_rag != st.session_state.use_rag:
                st.session_state.use_rag = use_rag
                
            # Chunking method selection (only show if RAG is enabled)
            if st.session_state.use_rag:
                chunking_methods = {
                    'standard': 'Standard Chunking',
                    'semantic': 'Semantic Chunking',
                    'hyde': 'HyDE Retrieval'
                }
                
                chunking_method = st.selectbox(
                    "Chunking/Retrieval Method",
                    options=list(chunking_methods.keys()),
                    format_func=lambda x: chunking_methods[x],
                    index=list(chunking_methods.keys()).index(st.session_state.chunking_method) if st.session_state.chunking_method in chunking_methods else 0,
                    help="Choose how documents are chunked and retrieved"
                )
                
                if chunking_method != st.session_state.chunking_method:
                    st.session_state.chunking_method = chunking_method
                    # Update HyDE toggle based on chunking method
                    st.session_state.use_hyde = (chunking_method == 'hyde')
                    # If semantic chunking is selected, we need to rebuild the vector store
                    if chunking_method == 'semantic':
                        with st.spinner("Rebuilding vector store with semantic chunking..."):
                            self.rag_system = self.initialize_vector_store("backend/docs", rebuild=True)
                        st.success("Vector store rebuilt with semantic chunking!")
                        st.rerun()
                
                # Show HyDE settings if HyDE is selected
                if st.session_state.chunking_method == 'hyde':
                    st.session_state.use_hyde = True
                    hyde_chunk_size = st.slider(
                        "HyDE chunk size", 
                        min_value=100, 
                        max_value=1000, 
                        value=st.session_state.hyde_chunk_size,
                        step=100,
                        help="Size of hypothetical document in characters"
                    )
                    if hyde_chunk_size != st.session_state.hyde_chunk_size:
                        st.session_state.hyde_chunk_size = hyde_chunk_size
                else:
                    st.session_state.use_hyde = False
                
                # Show semantic chunking settings if semantic chunking is selected
                if st.session_state.chunking_method == 'semantic':
                    with st.expander("Semantic Chunking Settings"):
                        breakpoint_types = {
                            'percentile': 'Percentile',
                            'standard_deviation': 'Standard Deviation',
                            'interquartile': 'Interquartile'
                        }
                        
                        breakpoint_type = st.selectbox(
                            "Breakpoint Type",
                            options=list(breakpoint_types.keys()),
                            format_func=lambda x: breakpoint_types[x],
                            index=list(breakpoint_types.keys()).index(st.session_state.semantic_breakpoint_type) if st.session_state.semantic_breakpoint_type in breakpoint_types else 0,
                            help="Method to determine where to split text"
                        )
                        
                        if breakpoint_type != st.session_state.semantic_breakpoint_type:
                            st.session_state.semantic_breakpoint_type = breakpoint_type
                            st.session_state.custom_config['semantic_breakpoint_type'] = breakpoint_type
                        
                        threshold_amount = st.slider(
                            "Threshold Amount", 
                            min_value=50, 
                            max_value=99,
                            value=st.session_state.semantic_threshold_amount,
                            step=5,
                            help="Higher values create fewer chunks (90 = 90th percentile)"
                        )
                        
                        if threshold_amount != st.session_state.semantic_threshold_amount:
                            st.session_state.semantic_threshold_amount = threshold_amount
                            st.session_state.custom_config['semantic_threshold_amount'] = threshold_amount
                        
                        # Add k value slider for semantic chunking
                        semantic_k = st.slider(
                            "Number of chunks to retrieve (k)", 
                            min_value=1, 
                            max_value=5, 
                            value=st.session_state.k,
                            step=1,
                            help="How many semantically similar chunks to retrieve per query"
                        )
                        
                        if semantic_k != st.session_state.k:
                            st.session_state.k = semantic_k
                            st.session_state.custom_config['k'] = semantic_k
                        
                        if st.button("Apply Semantic Chunking Settings"):
                            with st.spinner("Rebuilding vector store with new semantic chunking settings..."):
                                self.rag_system = self.initialize_vector_store("backend/docs", rebuild=True)
                            st.success("Vector store rebuilt with new semantic chunking settings!")
                            st.rerun()
                
            # Temperature slider
            # Ensure the value is a float before using it in the slider
            current_temp = st.session_state.temperature
            if not isinstance(current_temp, (int, float)):
                current_temp = 0.0
            else:
                current_temp = float(current_temp)
                
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=current_temp,
                step=0.1
            )
            if temperature != st.session_state.temperature:
                st.session_state.temperature = float(temperature)
                st.session_state.custom_config['temperature'] = float(temperature)
                self.config_manager.update_runtime_config('temperature', float(temperature))
                
            # k value for retrieval
            # Ensure the value is an integer before using it in the slider
            current_k = st.session_state.k
            if not isinstance(current_k, (int, float)):
                current_k = 1
            else:
                current_k = int(current_k)
                
            k_value = st.slider(
                "Number of documents to retrieve (k)", 
                min_value=1, 
                max_value=5, 
                value=current_k,
                step=1
            )
            if k_value != st.session_state.k:
                st.session_state.k = int(k_value)
                st.session_state.custom_config['k'] = int(k_value)
                self.config_manager.update_runtime_config('k', int(k_value))
                
            st.markdown("---")
            
            # Reset to defaults button
            if st.button("Reset to defaults"):
                try:
                    # Reset the config manager
                    reset_success = self.config_manager.reset_to_defaults()
                    if not reset_success:
                        st.error("Failed to reset configuration")
                        return
                        
                    # Get the fresh config
                    self.config = self.config_manager.get_config()
                    
                    # Reset session state values using stored defaults
                    st.session_state.provider = self.default_provider
                    st.session_state.model = self.default_model
                    st.session_state.messages[0]["content"] = self.default_system_message
                    st.session_state.temperature = self.default_temperature
                    st.session_state.k = self.default_k
                    st.session_state.custom_config = {}
                    
                    st.success("Settings reset to defaults")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error resetting to defaults: {str(e)}")
                    import traceback
                    st.exception(traceback.format_exc())

            # Clear chat button
            if st.button("Clear chat"):
                # Keep only the system message
                st.session_state.messages = [st.session_state.messages[0]]
                st.rerun()
                
            # Rebuild vector store button
            if st.button("Rebuild Vector Store"):
                with st.spinner("Rebuilding vector database from documents..."):
                    try:
                        # Force rebuilding by passing rebuild=True
                        self.rag_system = self.initialize_vector_store("backend/docs", rebuild=True)
                        st.success("Vector store rebuilt successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error rebuilding vector store: {str(e)}")
                
            st.markdown("---")
                
            # Show current configuration (for debugging)
            with st.expander("Configuration Debug Info"):
                st.subheader("Custom Settings")
                st.json(st.session_state.custom_config)
                
                st.subheader("Current Session Values")
                session_values = {
                    "provider": st.session_state.provider,
                    "model": st.session_state.model,
                    "temperature": st.session_state.temperature,
                    "k": st.session_state.k,
                    "use_rag": st.session_state.use_rag,
                    "chunking_method": st.session_state.chunking_method,
                    "use_hyde": st.session_state.use_hyde,
                    "hyde_chunk_size": st.session_state.hyde_chunk_size,
                    "semantic_breakpoint_type": st.session_state.semantic_breakpoint_type,
                    "semantic_threshold_amount": st.session_state.semantic_threshold_amount,
                    "system_message": st.session_state.messages[0]["content"] if len(st.session_state.messages) > 0 else "No system message"
                }
                st.json(session_values)
                
                st.subheader("Default Values")
                default_values = {
                    "provider": self.default_provider,
                    "model": self.default_model,
                    "temperature": self.default_temperature,
                    "k": self.default_k,
                    "system_message": self.default_system_message
                }
                st.json(default_values)

    def display_chat_messages(self):
        """Display existing chat messages"""
        # Display all messages except the system message
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    def handle_user_input(self):
        """Process user input and generate response"""
        # Get user input from chat input box
        prompt = st.chat_input(self.config.get('chat_placeholder', "Hi, how can I help you?"))

        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    # Initialize hypothetical_doc variable
                    hypothetical_doc = None
                    
                    if st.session_state.use_rag:
                        # Update RAG system if parameters changed
                        if (st.session_state.k != self.config.get('k') or
                            st.session_state.model != self.llm.model_name or
                            st.session_state.provider != st.session_state.provider or
                            st.session_state.temperature != self.llm.temperature):
                            
                            # Get system message if it exists
                            system_message = None
                            if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
                                system_message = st.session_state.messages[0]["content"]
                                
                            self.rag_system = update_rag_parameters(
                                self.rag_system,
                                provider=st.session_state.provider,
                                model=st.session_state.model,
                                temperature=st.session_state.temperature,
                                k=st.session_state.k,
                                system_message=system_message
                            )
                        
                        # Use HyDE retrieval if enabled
                        if st.session_state.use_hyde:
                            with st.spinner("Generating hypothetical document and searching..."):
                                # Get HyDE retriever
                                hyde_retriever = get_hyde_retriever(
                                    provider=st.session_state.provider,
                                    model=st.session_state.model,
                                    temperature=st.session_state.temperature,
                                    chunk_size=st.session_state.hyde_chunk_size
                                )
                                
                                # Retrieve documents
                                retrieval_result = hyde_retriever.retrieve(prompt, k=st.session_state.k)
                                retrieved_docs = retrieval_result["documents"]
                                hypothetical_doc = retrieval_result["hypothetical_doc"]
                                
                                # Format context for the LLM
                                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                                
                                # Create a prompt with the context
                                system_prompt = st.session_state.messages[0]["content"] if len(st.session_state.messages) > 0 else ""
                                
                                # Create messages for the LLM
                                if st.session_state.provider == 'openai' and st.session_state.model in ["o3-mini", "o1-mini"]:
                                    # For o3-mini and o1-mini, only use simple user messages
                                    messages = [
                                        {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
                                    ]
                                else:
                                    messages = [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
                                    ]
                                
                                # Get response from LLM
                                try:
                                    if st.session_state.provider == 'openai':
                                        if st.session_state.model in ["o3-mini", "o1-mini"]:
                                            llm = ChatOpenAI(model=st.session_state.model)
                                        else:
                                            llm = ChatOpenAI(model=st.session_state.model, temperature=st.session_state.temperature)
                                    else:
                                        llm = ChatAnthropic(model=st.session_state.model, temperature=st.session_state.temperature)
                                    
                                    llm_response = llm.invoke(messages)
                                    response = llm_response.content
                                except Exception:
                                    # If we encounter an error, try a more basic approach
                                    st.warning("Encountered an error with the model. Trying a simplified approach.")
                                    
                                    # Create a very simple message
                                    simple_messages = [{"role": "user", "content": f"Answer this question using the following context.\n\nContext: {context}\n\nQuestion: {prompt}"}]
                                    
                                    if st.session_state.provider == 'openai':
                                        llm = ChatOpenAI(model=st.session_state.model)
                                    else:
                                        llm = ChatAnthropic(model=st.session_state.model)
                                        
                                    llm_response = llm.invoke(simple_messages)
                                    response = llm_response.content
                                
                                # Add HyDE information to the response
                                source_info = "\n\n**Sources:**\n"
                                for i, doc in enumerate(retrieved_docs):
                                    title = doc.metadata.get('title', 'Unknown')
                                    page = doc.metadata.get('page', 'Unknown')
                                    source_info += f"{i+1}. {title} (Page {page})\n"
                                
                                response += source_info
                                
                                # Display the response
                                message_placeholder.markdown(response)
                                
                                # Store hypothetical_doc for later display
                                hypothetical_doc = retrieval_result["hypothetical_doc"]
                        else:
                            # Use standard RAG system for response
                            with st.spinner("Searching documents and generating response..."):
                                result = query_rag_chain(self.rag_system["rag_chain"], prompt)
                                response = result["answer"]
                                
                                # Show sources if available
                                if result["source_documents"]:
                                    source_info = "\n\n**Sources:**\n"
                                    for i, doc in enumerate(result["source_documents"]):
                                        title = doc.metadata.get('title', 'Unknown')
                                        page = doc.metadata.get('page', 'Unknown')
                                        source_info += f"{i+1}. {title} (Page {page})\n"
                                    response += source_info
                    else:
                        # Create or update the appropriate LLM based on provider
                        if st.session_state.provider == 'openai':
                            # Special handling for models that don't support temperature
                            if st.session_state.model in ["o3-mini", "o1-mini"]:
                                if (not isinstance(self.llm, ChatOpenAI) or 
                                    self.llm.model_name != st.session_state.model):
                                    self.llm = ChatOpenAI(
                                        model=st.session_state.model,
                                        streaming=True
                                    )
                            else:
                                if (not isinstance(self.llm, ChatOpenAI) or 
                                    self.llm.model_name != st.session_state.model or 
                                    self.llm.temperature != st.session_state.temperature):
                                    self.llm = ChatOpenAI(
                                        model=st.session_state.model,
                                        temperature=st.session_state.temperature,
                                        streaming=True
                                    )
                        else:  # Anthropic
                            if (not isinstance(self.llm, ChatAnthropic) or 
                                self.llm.model_name != st.session_state.model or 
                                self.llm.temperature != st.session_state.temperature):
                                try:
                                    # Try to create with system message if it exists
                                    if st.session_state.messages[0]["role"] == "system":
                                        system_content = st.session_state.messages[0]["content"]
                                        self.llm = ChatAnthropic(
                                            model=st.session_state.model,
                                            temperature=st.session_state.temperature,
                                            streaming=True,
                                            system=system_content
                                        )
                                    else:
                                        self.llm = ChatAnthropic(
                                            model=st.session_state.model,
                                            temperature=st.session_state.temperature,
                                            streaming=True
                                        )
                                except Exception:
                                    # If 'system' param fails, create without it
                                    self.llm = ChatAnthropic(
                                        model=st.session_state.model,
                                        temperature=st.session_state.temperature,
                                        streaming=True
                                    )
                        
                        # Format messages for LangChain - ensuring roles are compatible
                        messages = []
                        for m in st.session_state.messages:
                            # Skip the first system message - it's handled separately
                            if m["role"] == "system" and len(messages) == 0:
                                continue
                                
                            if st.session_state.provider == 'anthropic':
                                # Map 'user' to 'human' for Anthropic
                                role = 'human' if m["role"] == 'user' else m["role"]
                                if role == 'assistant':  # Keep 'assistant' role
                                    messages.append({"role": role, "content": m["content"]})
                                elif role == 'human':  # Use 'human' role
                                    messages.append({"role": role, "content": m["content"]})
                                # Skip any other roles (like 'system' if they appear later)
                            else:  # OpenAI
                                # For most OpenAI models, use standard roles
                                if m["role"] in ['user', 'assistant', 'system']:
                                    messages.append({"role": m["role"], "content": m["content"]})
                                    
                        # Ensure we're sending valid message types
                        if len(messages) == 0:
                            # If somehow we end up with no messages, add a dummy user message
                            messages.append({"role": "user" if st.session_state.provider == "openai" else "human", 
                                            "content": prompt})
                        
                        # Handle system message for Anthropic separately if needed
                        if st.session_state.provider == 'anthropic' and st.session_state.messages[0]["role"] == "system":
                            # For Anthropic, we'll use system in the messages constructor instead
                            # Different versions of langchain_anthropic may handle this differently
                            try:
                                # Try to create a new ChatAnthropic with the system message
                                system_content = st.session_state.messages[0]["content"]
                                self.llm = ChatAnthropic(
                                    model=st.session_state.model,
                                    temperature=st.session_state.temperature,
                                    streaming=True,
                                    system=system_content
                                )
                            except Exception:
                                # If that fails, fall back to prepending to the first human message
                                st.warning("Note: System prompt will be included with your first message for Claude")
                                if len(messages) > 0 and messages[0]["role"] == "human":
                                    system_content = st.session_state.messages[0]["content"]
                                    messages[0]["content"] = f"System: {system_content}\n\nHuman: {messages[0]['content']}"
                        
                        # Use LangChain with streaming
                        response_chunks = []
                        
                        # Create a placeholder for displaying the streaming text
                        for chunk in self.llm.stream(messages):
                            content = chunk.content
                            response_chunks.append(content)
                            message_placeholder.markdown("".join(response_chunks) + "▌")
                        
                        # Combine all chunks for the final response
                        response = "".join(response_chunks)

                    # Display the response
                    message_placeholder.markdown(response)
                    
                    # Add the hypothetical document expander if HyDE was used
                    if st.session_state.use_rag and st.session_state.use_hyde:
                        with st.expander("View Hypothetical Document Used for Retrieval"):
                            st.markdown("### Hypothetical Document")
                            st.markdown(hypothetical_doc)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_message = str(e)
                    # Extract the most relevant part of OpenAI errors
                    if "openai.BadRequestError" in error_message:
                        if "Unsupported parameter: 'temperature'" in error_message or "'temperature' does not support" in error_message:
                            error_message = f"Error: The model {st.session_state.model} doesn't support temperature parameter. Please try again with a different model or reset the app."
                            # Try to automatically fix the issue by recreating the model without temperature
                            try:
                                if st.session_state.provider == 'openai':
                                    self.llm = ChatOpenAI(model=st.session_state.model, streaming=True)
                                    # Also update the config to mark this model as not supporting temperature
                                    if st.session_state.model not in ["o3-mini", "o1-mini"]:
                                        st.warning(f"Adding {st.session_state.model} to the list of models that don't support temperature for future use.")
                                    message_placeholder.markdown("Attempting to reconnect with the model. Please try again...")
                            except Exception as fix_error:
                                st.error(f"Could not automatically fix the issue: {str(fix_error)}")
                        elif "Unsupported value: 'messages[0].role'" in error_message:
                            error_message = f"Error: The model {st.session_state.model} doesn't support one of the message roles. Please try again or switch models."
                        elif "maximum context length" in error_message.lower():
                            error_message = f"Error: The conversation is too long for the {st.session_state.model} model. Please clear the chat or use a model with longer context."
                        else:
                            # Show a more user-friendly error for other OpenAI errors
                            error_message = f"OpenAI API Error: {error_message.split('-')[-1].strip()}"
                    
                    # Show the error to the user
                    st.error(error_message)
                    
                    # Log the full error for debugging
                    import traceback
                    print(f"Full error: {traceback.format_exc()}")

    def run(self):
        """Run the Streamlit application"""
        st.title(self.config.get('app_title', "ChatBot"))

        # Display sidebar with settings
        self.display_sidebar()

        # Display existing chat messages
        self.display_chat_messages()

        # Handle user input
        self.handle_user_input()