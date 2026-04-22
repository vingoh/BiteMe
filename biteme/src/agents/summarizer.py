from src.memory.shared import GraphState


def summarizer_node(state: GraphState, llm) -> GraphState:
    answered_question_ids = {
        str(item.get("id"))
        for item in state["question_plan"]
        if item.get("status") == "answered" and item.get("id")
    }
    explored_topics = [
        item.get("topic", "")
        for item in state["question_plan"]
        if str(item.get("id")) in answered_question_ids and item.get("topic")
    ]
    key_findings = _collect_key_findings(state)
    next_directions = _collect_next_directions(state)

    summary: dict[str, object] = {
        "explored_topics": explored_topics,
        "key_findings": key_findings,
        "next_directions": next_directions,
    }

    if state["mode"] == "interview":
        summary["understanding_assessment"] = _build_understanding_assessment(
            state, answered_question_ids
        )

    state["final_summary"] = summary
    return state


def _collect_key_findings(state: GraphState) -> list[str]:
    findings: list[str] = []
    for entry in state["dialogue_history"]:
        if entry.get("role") in {"answerer", "user"} and entry.get("content"):
            findings.append(str(entry["content"]))

    if findings:
        return findings
    return ["No concrete findings were captured in dialogue history."]


def _collect_next_directions(state: GraphState) -> list[str]:
    pending_topics = [
        item.get("topic", "")
        for item in state["question_plan"]
        if item.get("status") == "pending" and item.get("topic")
    ]
    if pending_topics:
        return [f"Continue with: {topic}" for topic in pending_topics]
    return ["No pending topics; expand with deeper edge-case exploration."]


def _build_understanding_assessment(
    state: GraphState, answered_question_ids: set[str]
) -> dict[str, object]:
    evaluations = [
        str(entry.get("evaluation", ""))
        for entry in state["dialogue_history"]
        if entry.get("role") == "user" and entry.get("evaluation")
    ]
    return {
        "answered_questions": len(answered_question_ids),
        "evaluation_signals": evaluations,
    }
