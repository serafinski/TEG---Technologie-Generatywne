# Multiagents Implementations and MLOps using Langsmith

## Overview
Today we will continue working on the research assistant multiagent system. We will expand previous implementation with more agents, looping and stoping mechanisms and additional tools. Afterwards we'll explore how to introduce tracing and monitoring easily using LangSmith library. 

## Documentation
  - [LangChain Documentation](https://www.langchain.com/langchain)
  - [LangGraph Documentation](https://www.langchain.com/langgraph)
  - [LangSmith Documentation](https://docs.smith.langchain.com/)

## Useful materials
 - [LangSmith tutorial](https://www.youtube.com/watch?v=Hab2CV_0hpQ)
 - [How to evaluate RAGs using RAGAs with LangSmith](https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/)
  
## First steps
In order to run jupyter you need to create `.venv` and install all requirements.txt just like we did during all previous classes. 
Additionaly, get your Tavily API key from https://app.tavily.com/
Moreover, you will need a LangSmith API key obtainable after you register on https://www.langchain.com/langsmith.
Use your API keys in by renaming `.env - template` file to `.env` and filling in missing keys values. 

## Topics Covered in this Class
Multiagent Graph Workflows
Building and visualizing agent interaction graphs using LangGraph to manage complex workflows.
Role-specific Agents

Defining clear roles for agents such as:
- Search Agent: Performs web searches for relevant news.
- Outliner Agent: Structures collected information into coherent outlines.
- Writer Agent: Drafts comprehensive articles based on outlines.
- Editor Agent: Provides iterative feedback to improve article quality.

Conditional Logic & Iterative Feedback

Implementing loops and conditional routing between agents to ensure output refinement.

External Tools Integration

Enhancing agent functionality through tools such as Tavily for web search, and optionally Wikipedia and Arxiv for additional resources.

LLMOps and Monitoring with Langsmith

Utilizing Langsmith for real-time monitoring, debugging, and optimizing agent performance.