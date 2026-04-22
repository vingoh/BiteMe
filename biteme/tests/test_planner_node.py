from src.agents.planner import planner_node
from src.memory.shared import new_graph_state


def test_planner_initializes_question_plan_from_keywords() -> None:
    state = new_graph_state(project_content="simple project")
    state["user_keywords"] = ["architecture"]

    result = planner_node(state, llm=None)

    assert len(result["question_plan"]) >= 1
    assert result["question_plan"][0]["topic"] == "architecture"
    assert "id" in result["question_plan"][0]


def test_planner_uses_default_topics_when_keywords_empty() -> None:
    state = new_graph_state(project_content="simple project")

    result = planner_node(state, llm=None)

    topics = [item["topic"] for item in result["question_plan"]]
    assert topics == ["project-overview", "core-flow", "trade-offs"]


def test_planner_preserves_existing_question_plan() -> None:
    state = new_graph_state(project_content="simple project")
    state["question_plan"] = [
        {"id": "q-existing", "topic": "already-planned", "priority": "1", "status": "pending"}
    ]
    state["user_keywords"] = ["architecture"]

    result = planner_node(state, llm=None)

    assert result["question_plan"] == [
        {"id": "q-existing", "topic": "already-planned", "priority": "1", "status": "pending"}
    ]
