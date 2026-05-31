from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import SessionState
from .nodes import planner_node, questioner_node, answerer_node, reviewer_node


def _after_answerer(state: SessionState) -> str:
    if "answerer" in state["hitl_flags"]:
        return "reviewer"
    if state["turn_count"] >= state["max_turns"]:
        return END
    return "questioner"


def _should_continue(state: SessionState) -> str:
    if state["turn_count"] >= state["max_turns"]:
        return END
    return "questioner"


def build_graph(checkpointer: SqliteSaver) -> StateGraph:
    builder = StateGraph(SessionState)
    builder.add_node("planner", planner_node)
    builder.add_node("questioner", questioner_node)
    builder.add_node("answerer", answerer_node)
    builder.add_node("reviewer", reviewer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "questioner")
    builder.add_edge("questioner", "answerer")
    builder.add_conditional_edges(
        "answerer",
        _after_answerer,
        {"reviewer": "reviewer", "questioner": "questioner", END: END},
    )
    builder.add_conditional_edges(
        "reviewer",
        _should_continue,
        {"questioner": "questioner", END: END},
    )
    return builder.compile(checkpointer=checkpointer, interrupt_before=[])
