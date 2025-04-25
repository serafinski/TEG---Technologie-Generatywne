from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from agents.agent_state import AgentState

from langchain_core.messages import HumanMessage, AIMessage

class WorkflowBuilder:
    def __init__(self, agents, tools):
        self.agents = agents
        self.tools = tools
        self.workflow = StateGraph(AgentState)

    def _agent_node(self, agent, name: str):
        def node(state):
            if state.get("retry_count", 0) >= 3:
                return {"messages": [HumanMessage(content="DONE")]}  # Use proper message type
            
            # Preserve message types
            new_state = state.copy()
            response = agent.invoke(new_state)
            return {"messages": [response]}  # Expecting AIMessage/ToolMessage
        node.__name__ = name
        return node

    def _setup_nodes(self):
        nodes = {
            "search": self._agent_node(self.agents["search"], "Search Agent"),
            "outliner": self._agent_node(self.agents["outliner"], "Outliner Agent"),
            "writer": self._agent_node(self.agents["writer"], "Writer Agent"),
            "editor": self._agent_node(self.agents["editor"], "Editor Agent"),
            "tools": ToolNode(self.tools),
            "finalizer": lambda state: state  # Direct pass-through
        }
        
        for name, node in nodes.items():
            self.workflow.add_node(name, node)

    def _setup_edges(self):
        self.workflow.set_entry_point("search")
        
        # Search node transitions
        self.workflow.add_conditional_edges(
            "search",
            self._should_search,
            {"tools": "tools", "outliner": "outliner"}
        )
        
        # Tools node always returns to search
        self.workflow.add_edge("tools", "search")
        
        # Main workflow chain
        self.workflow.add_edge("outliner", "writer")
        self.workflow.add_edge("writer", "editor")
        
        # Editor decision point
        self.workflow.add_conditional_edges(
            "editor",
            self._should_edit,
            {"revise": "writer", "finalize": "finalizer"}
        )
        
        # Final connection to END
        self.workflow.add_edge("finalizer", END)

    def build(self):
        self._setup_nodes()
        self._setup_edges()
        return self.workflow.compile()

    def _should_search(self, state):
        state["retry_count"] = state.get("retry_count", 0)
        
        # Modified message checking
        recent_messages = [msg for msg in state["messages"][-3:] if isinstance(msg, AIMessage)]
        tool_calls_detected = any(msg.tool_calls for msg in recent_messages)
        
        if tool_calls_detected:
            state["retry_count"] += 1
            return "tools" if state["retry_count"] < 3 else "outliner"
        return "outliner"

    def _should_edit(self, state):
        content = state["messages"][-1].get("content", "")
        state["retry_count"] = state.get("retry_count", 0)
        
        if "DONE" in content or state["retry_count"] >= 2:
            state["retry_count"] = 0
            return "finalize"
        
        state["retry_count"] += 1
        return "revise"
