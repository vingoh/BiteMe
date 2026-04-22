from src import cli


def test_run_cli_session_executes_orchestrator_flow() -> None:
    result = cli.run_cli_session(
        project_content="demo",
        user_keywords=["architecture", "testing"],
        mode="observe",
        max_rounds=5,
    )

    assert result["terminated"] is True
    assert result["termination_reason"] == "plan_completed"
    assert result["completed_question_ids"] == ["q1", "q2"]
    assert result["final_summary"]["explored_topics"] == ["architecture", "testing"]


def test_run_cli_uses_helper_and_prints_completion(monkeypatch) -> None:
    inputs = iter(["sample project", "architecture,testing", "", ""])
    printed: list[str] = []

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(cli.Console, "print", lambda _self, message: printed.append(str(message)))

    cli.run_cli()

    assert any("BiteMe CLI started." in line for line in printed)
    assert any("Session finished." in line for line in printed)
    assert any("plan_completed" in line for line in printed)


def test_run_cli_mode_selection_passes_interview(monkeypatch) -> None:
    inputs = iter(["sample project", "architecture,testing", "interview", "", "my answer"])
    captured: dict[str, object] = {}

    def fake_run_cli_session(**kwargs):
        captured.update(kwargs)
        return {
            "termination_reason": "plan_completed",
            "completed_question_ids": ["q1"],
            "final_summary": {"explored_topics": ["architecture"]},
        }

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(cli, "run_cli_session", fake_run_cli_session)
    monkeypatch.setattr(cli.Console, "print", lambda _self, _message: None)

    cli.run_cli()

    assert captured["mode"] == "interview"
    assert captured["interview_answer"] == "my answer"


def test_run_cli_routes_intervention_commands_through_orchestrator(monkeypatch) -> None:
    inputs = iter(["sample project", "", "observe", "/ask architecture", "/skip", "/stop"])
    captured: dict[str, object] = {}

    def fake_run_cli_session(**kwargs):
        captured.update(kwargs)
        return {
            "termination_reason": "plan_completed",
            "completed_question_ids": ["q1"],
            "final_summary": {},
        }

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(cli, "run_cli_session", fake_run_cli_session)
    monkeypatch.setattr(cli.Console, "print", lambda _self, _message: None)

    cli.run_cli()

    assert captured["intervention_commands"] == ["/ask architecture", "/skip", "/stop"]


def test_run_cli_session_applies_interventions_and_interview_answer(monkeypatch) -> None:
    captured_state: dict[str, object] = {}

    monkeypatch.setattr(cli, "build_graph", lambda: object())

    def fake_run_session(_graph, state):
        captured_state.update(state)
        return state

    monkeypatch.setattr(cli, "run_session", fake_run_session)

    cli.run_cli_session(
        project_content="demo",
        user_keywords=["architecture"],
        mode="interview",
        intervention_commands=["/ask extra topic"],
        interview_answer="user explanation",
    )

    assert captured_state["mode"] == "interview"
    assert captured_state["user_answer"] == "user explanation"
    assert captured_state["question_plan"][0]["topic"] == "extra topic"
