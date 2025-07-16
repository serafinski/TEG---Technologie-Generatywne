# TEG-RAG: Travel Document Retrieval-Augmented Generation

This application is a Retrieval-Augmented Generation (RAG) system that allows users to query travel-related documents. It uses a combination of vector embeddings and large language models to provide accurate, context-aware responses based on the content of travel brochures and company information.

*Side Note: Not my best work - didn't have much time to work on this.*

## Features

- Interactive chat interface built with Streamlit
- Support for multiple LLM providers (OpenAI and Anthropic)
- Multiple document chunking strategies:
  - Standard chunking
  - Semantic chunking
  - Hypothetical Document Embeddings (HyDE)
- Configurable retrieval parameters
- PDF document processing
- Vector database storage for efficient retrieval

## Requirements

- Python 3.11 or higher
- OpenAI API key and/or Anthropic API key

## Installation

### Using pip

1. Clone the repository:
   ```
   git clone <repository-url>
   cd teg-rag
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Using conda

1. Clone the repository:
   ```
   git clone <repository-url>
   cd teg-rag
   ```

2. Create a conda environment:
   ```
   conda create -n teg-rag python=3.11
   conda activate teg-rag
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Environment Setup

1. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Configuration

The application is configured through the `config.yaml` file. You can modify:

- Default LLM provider and model
- Available models for each provider
- System message for the chatbot
- Default temperature and retrieval parameters
- Chunking method and settings

## Running the Application

Start the application with:

```
streamlit run app.py
```

This will launch a Streamlit web interface, typically accessible at http://localhost:8501

## Usage

1. The main chat interface allows you to ask questions about the travel documents
2. Use the sidebar to configure:
   - LLM provider (OpenAI or Anthropic)
   - Model selection
   - System message
   - RAG settings (chunking method, temperature, etc.)
   - Rebuild the vector database if needed

## Project Structure

- `app.py`: Main application entry point
- `config_manager.py`: Manages application configuration
- `config.yaml`: Configuration settings
- `frontend/`: Contains the Streamlit UI components
- `backend/`: Contains the RAG implementation
  - `chain.py`: LangChain setup for RAG
  - `hyde.py`: Hypothetical Document Embeddings implementation
  - `pdf_loader.py`: PDF document processing
  - `rag.py`: Main RAG system implementation
  - `semantic_chunking.py`: Semantic document chunking
  - `splitter.py`: Document splitting utilities
  - `vector_store.py`: Vector database management
  - `docs/`: Sample travel documents

## Customization

- Add your own PDF documents to the `backend/docs/` directory
- Modify the system message in `config.yaml` to change the chatbot's behavior
- Adjust chunking parameters for different document types

## Troubleshooting

- If you encounter errors related to the vector database, try using the "Rebuild Vector Store" button in the sidebar
- Ensure your API keys are correctly set in the `.env` file
- Check that you're using Python 3.11 or higher