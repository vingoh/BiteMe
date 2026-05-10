from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from .state import SessionState
from .nodes import questioner_node, answerer_node


def _should_continue(state: SessionState) -> str:
    if state["turn_count"] >= state["max_turns"]:
        return END
    return "continue"


def build_graph(checkpointer: BaseCheckpointSaver) -> StateGraph:
    builder = StateGraph(SessionState)
    builder.add_node("questioner", questioner_node)
    builder.add_node("answerer", answerer_node)

    builder.set_entry_point("questioner")
    builder.add_edge("questioner", "answerer")
    builder.add_conditional_edges(
        "answerer",
        _should_continue,
        {"continue": "questioner", END: END},
    )
    return builder.compile(checkpointer=checkpointer, interrupt_before=[])
