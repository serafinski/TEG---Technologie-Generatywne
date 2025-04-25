from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from managers.prompt_manager import PromptManager

class AgentFactory:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.prompt_manager = PromptManager()

    def create_agent(self, agent_type: str) -> Runnable:
        system_message = self.prompt_manager.get_template(agent_type)
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", "{system_message}"
                ),
                MessagesPlaceholder(variable_name="messages")
            ]).partial(system_message=system_message)
        
        if agent_type == "search":
            return prompt | self.llm.bind_tools(self.tools)
        return prompt | self.llm
