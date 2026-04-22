from typing import Optional

from src.memory.shared import GraphState


def _next_pending_question(state: GraphState) -> Optional[dict[str, str]]:
    for item in state["question_plan"]:
        if item.get("status") == "pending":
            return item
    return None


def questioner_node(state: GraphState, llm) -> GraphState:
    """Produce a question from the next pending plan item."""
    pending = _next_pending_question(state)
    if pending is None:
        return state

    question_id = pending["id"]
    topic = pending["topic"]
    question_text = f"Could you explain {topic}?"

    state["dialogue_history"].append(
        {"role": "questioner", "question_id": question_id, "content": question_text}
    )
    pending["status"] = "asked"
    return state
