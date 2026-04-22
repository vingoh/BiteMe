from src.memory.shared import GraphState

DEFAULT_TOPICS = ["project-overview", "core-flow", "trade-offs"]


def planner_node(state: GraphState, llm) -> GraphState:
    """Initialize a question plan from user keywords when one is missing."""
    if state["question_plan"]:
        return state

    seed_topics = state["user_keywords"] or DEFAULT_TOPICS
    state["question_plan"] = [
        {"id": f"q{i + 1}", "topic": topic, "priority": str(i + 1), "status": "pending"}
        for i, topic in enumerate(seed_topics)
    ]
    return state
