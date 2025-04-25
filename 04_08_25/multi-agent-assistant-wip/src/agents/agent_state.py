from typing import TypedDict

from typing import List


class AgentState(TypedDict):
    messages: List
    retry_count: int