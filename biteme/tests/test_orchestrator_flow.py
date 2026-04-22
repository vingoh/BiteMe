from src.memory.shared import new_graph_state
from src.orchestrator import apply_user_intervention, build_graph, run_session


def test_session_stops_when_plan_complete() -> None:
    graph = build_graph()
    state = new_graph_state(project_content="demo", max_rounds=10)
    state["user_keywords"] = ["architecture"]

    result = run_session(graph, state)

    assert result["terminated"] is True
    assert result["termination_reason"] in {"plan_completed", "max_rounds"}
    assert result["final_summary"]["explored_topics"] == ["architecture"]
    assert isinstance(result["final_summary"]["key_findings"], list)
    assert isinstance(result["final_summary"]["next_directions"], list)
    assert result["current_round"] == 1
    assert len(result["completed_question_ids"]) == 1
    assert any(entry["role"] == "questioner" for entry in result["dialogue_history"])
    assert any(entry["role"] == "answerer" for entry in result["dialogue_history"])


def test_run_session_terminates_when_max_rounds_reached() -> None:
    graph = build_graph()
    state = new_graph_state(project_content="demo", max_rounds=1)
    state["user_keywords"] = ["a", "b", "c"]

    result = run_session(graph, state)

    assert result["terminated"] is True
    assert result["termination_reason"] == "max_rounds"
    assert result["current_round"] == 1
    assert result["question_plan"][0]["status"] == "answered"
    assert any(item["status"] == "pending" for item in result["question_plan"][1:])


def test_interview_user_answer_consumed_once_across_rounds() -> None:
    graph = build_graph()
    state = new_graph_state(project_content="demo", mode="interview", max_rounds=10)
    state["user_keywords"] = ["architecture", "testing"]
    state["user_answer"] = "first interview answer"

    result = run_session(graph, state)

    user_entries = [entry for entry in result["dialogue_history"] if entry["role"] == "user"]
    answerer_entries = [
        entry for entry in result["dialogue_history"] if entry["role"] == "answerer"
    ]

    assert len(result["completed_question_ids"]) == 2
    assert len(user_entries) == 1
    assert user_entries[0]["content"] == "first interview answer"
    assert len(answerer_entries) == 1
    assert answerer_entries[0]["question_id"] == "q2"
    assert answerer_entries[0]["content"] == "answer: Could you explain testing?"


def test_apply_user_intervention_insert_question() -> None:
    state = new_graph_state(project_content="demo")

    updated = apply_user_intervention(state, "/ask explain architecture")

    assert updated["question_plan"][0]["topic"] == "explain architecture"


def test_apply_user_intervention_multiple_ask_generate_unique_ids() -> None:
    state = new_graph_state(project_content="demo")

    apply_user_intervention(state, "/ask first inserted")
    apply_user_intervention(state, "/ask second inserted")

    first_id = state["question_plan"][0]["id"]
    second_id = state["question_plan"][1]["id"]

    assert state["question_plan"][0]["topic"] == "second inserted"
    assert state["question_plan"][1]["topic"] == "first inserted"
    assert first_id != second_id

    apply_user_intervention(state, "/skip")
    apply_user_intervention(state, "/skip")

    assert first_id in state["completed_question_ids"]
    assert second_id in state["completed_question_ids"]
    assert len({first_id, second_id}) == 2


def test_apply_user_intervention_stop_marks_terminated() -> None:
    state = new_graph_state(project_content="demo")

    updated = apply_user_intervention(state, "/stop")

    assert updated["terminated"] is True
    assert updated["termination_reason"] == "user_stopped"


def test_apply_user_intervention_skip_marks_first_pending() -> None:
    state = new_graph_state(project_content="demo")
    state["question_plan"] = [
        {"id": "q1", "topic": "done", "priority": "1", "status": "answered"},
        {"id": "q2", "topic": "skip me", "priority": "2", "status": "pending"},
        {"id": "q3", "topic": "still pending", "priority": "3", "status": "pending"},
    ]

    updated = apply_user_intervention(state, "/skip")

    assert updated["question_plan"][1]["status"] == "skipped"
    assert updated["question_plan"][2]["status"] == "pending"
    assert "q2" in updated["completed_question_ids"]


def test_interview_mode_final_summary_contains_understanding_assessment() -> None:
    graph = build_graph()
    state = new_graph_state(project_content="demo", mode="interview", max_rounds=10)
    state["user_keywords"] = ["architecture"]
    state["user_answer"] = "I can explain the module boundaries."

    result = run_session(graph, state)

    assert "understanding_assessment" in result["final_summary"]
    assessment = result["final_summary"]["understanding_assessment"]
    assert assessment["answered_questions"] == 1
    assert assessment["evaluation_signals"] == ["placeholder-score"]


def test_run_session_observer_flow_completes_and_summarizes_all_topics() -> None:
    graph = build_graph()
    state = new_graph_state(project_content="demo", mode="observe", max_rounds=5)
    state["user_keywords"] = ["architecture", "testing"]

    result = run_session(graph, state)

    assert result["terminated"] is True
    assert result["termination_reason"] == "plan_completed"
    assert result["current_round"] == 2
    assert result["completed_question_ids"] == ["q1", "q2"]
    assert [item["status"] for item in result["question_plan"]] == ["answered", "answered"]
    assert result["final_summary"]["explored_topics"] == ["architecture", "testing"]
    assert result["final_summary"]["next_directions"] == ["No pending topics; expand with deeper edge-case exploration."]


def test_run_session_accepts_callable_graph_runner() -> None:
    state = new_graph_state(project_content="demo")

    def callable_graph(initial_state):
        next_state = dict(initial_state)
        next_state["terminated"] = True
        next_state["termination_reason"] = "callable_runner"
        return next_state

    result = run_session(callable_graph, state)

    assert result["terminated"] is True
    assert result["termination_reason"] == "callable_runner"
