from typing import Protocol

from src.memory.shared import GraphState


class AgentNode(Protocol):
    def __call__(self, state: GraphState, llm) -> GraphState:
        ...
