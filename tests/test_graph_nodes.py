import pytest
from unittest.mock import MagicMock, patch
from biteme.graph.state import SessionState, Turn
from biteme.graph.nodes import questioner_node, answerer_node
from biteme.graph.prompts import get_prompts

def test_session_state_has_outline_field():
    state = make_state(outline=["Q1", "Q2"])
    assert state["outline"] == ["Q1", "Q2"]

def test_session_state_outline_defaults_to_empty():
    state = make_state()
    assert state["outline"] == []

def make_state(**kwargs) -> SessionState:
    defaults = dict(
        mode="learn",
        messages=[],
        current_speaker="questioner",
        hitl_flags=[],
        turn_count=0,
        max_turns=10,
        context_strategy="direct",
        source_path="/tmp/fake",
        outline=[],
    )
    defaults.update(kwargs)
    return defaults  # type: ignore

def test_questioner_node_appends_turn(tmp_path):
    state = make_state(source_path=str(tmp_path))
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "What does foo do?"

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.get_overview.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        result = questioner_node(state)

    mock_provider.get_overview.assert_called_once()  # 第一轮走 get_overview，不走 retrieve
    mock_provider.retrieve.assert_not_called()
    assert result["current_speaker"] == "answerer"
    assert result["turn_count"] == 1
    assert len(result["messages"]) == 1
    assert result["messages"][0]["speaker"] == "questioner"

def test_answerer_node_always_retrieves(tmp_path):
    state = make_state(
        source_path=str(tmp_path),
        current_speaker="answerer",
        messages=[{"speaker": "questioner", "content": "What does foo do?", "retrieved_chunks": []}],
    )
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "foo returns None."

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.retrieve.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        result = answerer_node(state)

    mock_provider.retrieve.assert_called_once()   # 必须调用 retrieve
    assert result["current_speaker"] == "questioner"
    assert result["messages"][-1]["speaker"] == "answerer"
    assert len(result["messages"][-1]["retrieved_chunks"]) > 0


def test_get_prompts_learn_has_planner():
    prompts = get_prompts("learn")
    assert "planner" in prompts
    assert isinstance(prompts["planner"], str)
    assert len(prompts["planner"]) > 0

def test_get_prompts_interview_has_planner():
    prompts = get_prompts("interview")
    assert "planner" in prompts
    assert isinstance(prompts["planner"], str)
    assert len(prompts["planner"]) > 0
