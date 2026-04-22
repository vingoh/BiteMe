from typing import TypedDict


class RequiredGraphStateFields(TypedDict):
    project_content: str
    dialogue_history: list[dict[str, str]]
    question_plan: list[dict[str, str]]
    completed_question_ids: list[str]
    user_keywords: list[str]
    mode: str
    terminated: bool
    max_rounds: int
    current_round: int


class OptionalGraphStateFields(TypedDict, total=False):
    termination_reason: str
    final_summary: dict[str, object]
    user_answer: str


class GraphState(RequiredGraphStateFields, OptionalGraphStateFields):
    pass


def new_graph_state(
    project_content: str, *, mode: str = "observe", max_rounds: int = 8
) -> GraphState:
    """Create a fresh graph state with Task 1 default fields populated."""
    return {
        "project_content": project_content,
        "dialogue_history": [],
        "question_plan": [],
        "completed_question_ids": [],
        "user_keywords": [],
        "mode": mode,
        "terminated": False,
        "max_rounds": max_rounds,
        "current_round": 0,
    }
