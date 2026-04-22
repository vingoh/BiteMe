from typing import Optional

from src.memory.shared import GraphState


def _last_question(state: GraphState) -> Optional[dict[str, str]]:
    for item in reversed(state["dialogue_history"]):
        if item.get("role") == "questioner":
            return item
    return None


def _reference_answer(question: str) -> str:
    return f"A concise explanation for '{question}'."


def answerer_node(state: GraphState, llm, user_answer: Optional[str]) -> GraphState:
    """Append answerer output, handling observe and interview modes."""
    question_entry = _last_question(state)
    if question_entry is None:
        return state

    question_id = question_entry.get("question_id", "")
    question_text = question_entry["content"]

    if state["mode"] == "interview" and user_answer is not None:
        state["dialogue_history"].append(
            {
                "role": "user",
                "question_id": question_id,
                "content": user_answer,
                "reference_answer": _reference_answer(question_text),
                "evaluation": "placeholder-score",
            }
        )
        _mark_question_answered(state, question_id)
        return state

    state["dialogue_history"].append(
        {
            "role": "answerer",
            "question_id": question_id,
            "content": f"answer: {question_text}",
        }
    )
    _mark_question_answered(state, question_id)
    return state


def _mark_question_answered(state: GraphState, question_id: str) -> None:
    for item in state["question_plan"]:
        if item.get("id") == question_id:
            item["status"] = "answered"
            break

    if question_id and question_id not in state["completed_question_ids"]:
        state["completed_question_ids"].append(question_id)
    state["current_round"] += 1
