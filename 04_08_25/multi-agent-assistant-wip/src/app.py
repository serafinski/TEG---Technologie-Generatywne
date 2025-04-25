import argparse
import logging
import uuid
from managers.config_manager import Config
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from langchain_core.messages import HumanMessage

from workflow import WorkflowBuilder

from agents.agent_factory import AgentFactory

def main():
    config_obj = Config()
    llm = ChatOpenAI(
        openai_api_key=config_obj.key,
        model=config_obj.model,
        temperature=config_obj.temperature
    )
    
    # Define tools (for example, a search tool).
    tools = [TavilySearchResults(max_results=5)]
    agent_factory = AgentFactory(llm, tools)
    
    agents = {
        "search": agent_factory.create_agent("search"),
        "outliner": agent_factory.create_agent("outliner"),
        "writer": agent_factory.create_agent("writer"),
        "editor": agent_factory.create_agent("editor")
    }
    
    # Build the workflow.
    workflow = WorkflowBuilder(agents, tools).build()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    
    # Prepare initial state (note retry_count initialized to 0).
    inputs = {
        "messages": [HumanMessage(content=args.query)],
        "retry_count": 0
    }
    runnable_config = RunnableConfig(
        recursion_limit=100,
        metadata={
            "run_id": str(uuid.uuid4()),
            "max_retries": 3
        }
    )
    
    try:
        result = None
        last_output = None
        for output in workflow.stream(inputs, config=runnable_config, stream_mode="values"):
            logging.info(f"State update: {output}")
            last_output = output
        
        if last_output:
            result = last_output
    except GraphRecursionError as e:
        logging.error(f"Workflow terminated: {str(e)}")
        if last_output:
            result = last_output
    
    # Print the final message content.
    if result:
        print(result["messages"][-1]["content"])
    else:
        print("No result returned.")

if __name__ == "__main__":
    main()
