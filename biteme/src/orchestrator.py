from typing import Any, Callable

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight local envs
    END = "__end__"
    START = "__start__"
    StateGraph = None

from src.agents.answerer import answerer_node
from src.agents.planner import planner_node
from src.agents.questioner import questioner_node
from src.agents.summarizer import summarizer_node
from src.memory.shared import GraphState


def build_orchestrator_graph():
    if StateGraph is None:
        return _build_fallback_graph()

    workflow = StateGraph(GraphState)
    workflow.add_node("planner", _planner_step)
    workflow.add_node("questioner", _questioner_step)
    workflow.add_node("answerer", _answerer_step)
    workflow.add_node("summarize", _summarize_step)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"questioner": "questioner", "summarize": "summarize"},
    )
    workflow.add_edge("questioner", "answerer")
    workflow.add_edge("answerer", "planner")
    workflow.add_edge("summarize", END)
    return workflow.compile()


def build_graph():
    return build_orchestrator_graph()


def run_session(graph, initial_state: GraphState) -> GraphState:
    if hasattr(graph, "invoke"):
        return graph.invoke(initial_state)
    if callable(graph):
        return graph(initial_state)
    raise TypeError("graph must provide invoke(state) or be callable")


def apply_user_intervention(state: GraphState, command: str) -> GraphState:
    if command.startswith("/ask "):
        topic = command.replace("/ask ", "", 1).strip()
        if topic:
            state["question_plan"].insert(
                0,
                {
                    "id": _next_user_inserted_question_id(state),
                    "topic": topic,
                    "priority": "0",
                    "status": "pending",
                },
            )
    elif command.startswith("/skip"):
        for item in state["question_plan"]:
            if item.get("status") != "pending":
                continue

            item["status"] = "skipped"
            question_id = item.get("id")
            if question_id and question_id not in state["completed_question_ids"]:
                state["completed_question_ids"].append(question_id)
            break
    elif command.startswith("/stop"):
        state["terminated"] = True
        state["termination_reason"] = "user_stopped"
    return state


def _next_user_inserted_question_id(state: GraphState) -> str:
    prefix = "user-inserted-"
    existing_ids = {str(item.get("id")) for item in state["question_plan"] if item.get("id")}

    counter = 1
    while f"{prefix}{counter}" in existing_ids:
        counter += 1
    return f"{prefix}{counter}"


def _planner_step(state: GraphState) -> GraphState:
    return planner_node(state, llm=None)


def _questioner_step(state: GraphState) -> GraphState:
    return questioner_node(state, llm=None)


def _answerer_step(state: GraphState) -> GraphState:
    # Consume one queued interview answer at most once to avoid stale reuse.
    return answerer_node(state, llm=None, user_answer=state.pop("user_answer", None))


def _route_after_planner(state: GraphState) -> str:
    if state["current_round"] >= state["max_rounds"]:
        state["terminated"] = True
        state["termination_reason"] = "max_rounds"
        return "summarize"

    has_pending = any(item.get("status") == "pending" for item in state["question_plan"])
    if not has_pending:
        state["terminated"] = True
        state["termination_reason"] = "plan_completed"
        return "summarize"
    return "questioner"


def _summarize_step(state: GraphState) -> GraphState:
    return summarizer_node(state, llm=None)


class _FallbackCompiledGraph:
    def __init__(self, steps: dict[str, Callable[[GraphState], GraphState]]) -> None:
        self._steps = steps

    def invoke(self, state: GraphState) -> GraphState:
        next_node = "planner"
        while next_node != END:
            state = self._steps[next_node](state)
            if next_node == "planner":
                next_node = _route_after_planner(state)
            elif next_node == "questioner":
                next_node = "answerer"
            elif next_node == "answerer":
                next_node = "planner"
            else:
                next_node = END
        return state


def _build_fallback_graph() -> _FallbackCompiledGraph:
    return _FallbackCompiledGraph(
        {
            "planner": _planner_step,
            "questioner": _questioner_step,
            "answerer": _answerer_step,
            "summarize": _summarize_step,
        }
    )
