from src.agents.answerer import answerer_node
from src.agents.questioner import questioner_node
from src.agents.summarizer import summarizer_node
from src.memory.shared import new_graph_state


def test_questioner_uses_pending_plan_and_appends_to_history() -> None:
    state = new_graph_state(project_content="demo")
    state["question_plan"] = [
        {"id": "q1", "topic": "architecture", "priority": "1", "status": "pending"},
        {"id": "q2", "topic": "trade-offs", "priority": "2", "status": "pending"},
    ]

    result = questioner_node(state, llm=None)

    assert len(result["dialogue_history"]) == 1
    assert result["dialogue_history"][0]["role"] == "questioner"
    assert result["dialogue_history"][0]["question_id"] == "q1"
    assert "architecture" in result["dialogue_history"][0]["content"]


def test_answerer_appends_response_in_observe_mode() -> None:
    state = new_graph_state(project_content="demo", mode="observe")
    state["dialogue_history"].append(
        {"role": "questioner", "question_id": "q1", "content": "What is the architecture?"}
    )

    result = answerer_node(state, llm=None, user_answer=None)

    assert result["dialogue_history"][-1]["role"] == "answerer"
    assert "What is the architecture?" in result["dialogue_history"][-1]["content"]


def test_answerer_interview_mode_preserves_user_answer_and_adds_reference() -> None:
    state = new_graph_state(project_content="demo", mode="interview")
    state["dialogue_history"].append(
        {"role": "questioner", "question_id": "q1", "content": "Explain the architecture."}
    )

    result = answerer_node(state, llm=None, user_answer="It uses modules and interfaces.")

    assert result["dialogue_history"][-1] == {
        "role": "user",
        "question_id": "q1",
        "content": "It uses modules and interfaces.",
        "reference_answer": "A concise explanation for 'Explain the architecture.'.",
        "evaluation": "placeholder-score",
    }


def test_answerer_interview_mode_accepts_empty_user_answer() -> None:
    state = new_graph_state(project_content="demo", mode="interview")
    state["dialogue_history"].append(
        {"role": "questioner", "question_id": "q1", "content": "Explain the architecture."}
    )

    result = answerer_node(state, llm=None, user_answer="")

    assert result["dialogue_history"][-1] == {
        "role": "user",
        "question_id": "q1",
        "content": "",
        "reference_answer": "A concise explanation for 'Explain the architecture.'.",
        "evaluation": "placeholder-score",
    }


def test_summarizer_interview_mode_adds_task7_summary_structure() -> None:
    state = new_graph_state(project_content="demo", mode="interview")
    state["question_plan"] = [
        {"id": "q1", "topic": "architecture", "priority": "1", "status": "answered"},
        {"id": "q2", "topic": "testing", "priority": "2", "status": "pending"},
    ]
    state["completed_question_ids"] = ["q1"]
    state["dialogue_history"] = [
        {
            "role": "user",
            "question_id": "q1",
            "content": "It uses layered modules.",
            "evaluation": "placeholder-score",
        }
    ]

    result = summarizer_node(state, llm=None)

    assert "final_summary" in result
    summary = result["final_summary"]
    assert isinstance(summary, dict)
    assert set(summary.keys()) == {
        "explored_topics",
        "key_findings",
        "next_directions",
        "understanding_assessment",
    }
    assert isinstance(summary["explored_topics"], list)
    assert isinstance(summary["key_findings"], list)
    assert isinstance(summary["next_directions"], list)
    assert summary["explored_topics"] == ["architecture"]
    assert summary["key_findings"] == ["It uses layered modules."]
    assert summary["next_directions"] == ["Continue with: testing"]

    assessment = summary["understanding_assessment"]
    assert isinstance(assessment, dict)
    assert set(assessment.keys()) == {"answered_questions", "evaluation_signals"}
    assert isinstance(assessment["answered_questions"], int)
    assert isinstance(assessment["evaluation_signals"], list)
    assert assessment["answered_questions"] == 1
    assert assessment["evaluation_signals"] == ["placeholder-score"]


def test_summarizer_excludes_skipped_topics_from_explored_topics() -> None:
    state = new_graph_state(project_content="demo", mode="interview")
    state["question_plan"] = [
        {"id": "q1", "topic": "architecture", "priority": "1", "status": "answered"},
        {"id": "q2", "topic": "testing", "priority": "2", "status": "skipped"},
    ]
    state["completed_question_ids"] = ["q1", "q2"]
    state["dialogue_history"] = [
        {
            "role": "user",
            "question_id": "q1",
            "content": "Layered design.",
            "evaluation": "placeholder-score",
        }
    ]

    result = summarizer_node(state, llm=None)

    assert result["final_summary"]["explored_topics"] == ["architecture"]


def test_summarizer_counts_only_answered_questions_in_assessment() -> None:
    state = new_graph_state(project_content="demo", mode="interview")
    state["question_plan"] = [
        {"id": "q1", "topic": "architecture", "priority": "1", "status": "answered"},
        {"id": "q2", "topic": "testing", "priority": "2", "status": "skipped"},
    ]
    state["completed_question_ids"] = ["q1", "q2"]
    state["dialogue_history"] = [
        {
            "role": "user",
            "question_id": "q1",
            "content": "Layered design.",
            "evaluation": "placeholder-score",
        }
    ]

    result = summarizer_node(state, llm=None)

    assessment = result["final_summary"]["understanding_assessment"]
    assert assessment["answered_questions"] == 1
