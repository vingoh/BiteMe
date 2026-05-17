"""Real LangGraph + interrupt(); mocks only LLM and external APIs."""

from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from biteme.graph.graph import build_graph


def test_questioner_hitl_stream_emits_interrupt_dict(tmp_path):
    (tmp_path / "a.py").write_text("def foo(): pass")

    saver = InMemorySaver()
    graph = build_graph(checkpointer=saver)
    initial_state = {
        "mode": "learn",
        "messages": [],
        "current_speaker": "questioner",
        "hitl_flags": ["questioner"],
        "turn_count": 0,
        "max_turns": 3,
        "context_strategy": "direct",
        "source_path": str(tmp_path),
    }
    config = {"configurable": {"thread_id": "hitl-integration-1"}}

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "It returns None."

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm):
        saw_interrupt = False
        for state in graph.stream(
            initial_state, config=config, stream_mode="values"
        ):
            if isinstance(state, dict) and "__interrupt__" in state:
                saw_interrupt = True
                break

    assert saw_interrupt, "HITL questioner must yield values chunk with __interrupt__"


def test_questioner_hitl_resume_then_answerer_runs(tmp_path):
    (tmp_path / "a.py").write_text("def foo(): pass")

    saver = InMemorySaver()
    graph = build_graph(checkpointer=saver)
    initial_state = {
        "mode": "learn",
        "messages": [],
        "current_speaker": "questioner",
        "hitl_flags": ["questioner"],
        "turn_count": 0,
        "max_turns": 3,
        "context_strategy": "direct",
        "source_path": str(tmp_path),
    }
    config = {"configurable": {"thread_id": "hitl-integration-2"}}

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "The function returns None."

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm):
        # First pass: stop at interrupt
        pending = initial_state
        for state in graph.stream(pending, config=config, stream_mode="values"):
            if isinstance(state, dict) and "__interrupt__" in state:
                pending = Command(resume="What does foo do?")
                break
        else:
            pytest.fail("expected interrupt before resume")

        # Second pass: human question recorded, answerer runs
        final_messages = None
        for state in graph.stream(pending, config=config, stream_mode="values"):
            if isinstance(state, dict) and "messages" in state and state.get("messages"):
                final_messages = state["messages"]

    assert final_messages is not None
    assert final_messages[-1]["speaker"] in ("answerer", "human")
    # human question then answerer reply
    speakers = [m["speaker"] for m in final_messages]
    assert "human" in speakers
    assert "answerer" in speakers


def test_user_input_flows_through_to_answerer(tmp_path):
    """模拟人类输入，验证输入真的被传到 answerer 的 retrieve query。"""
    (tmp_path / "a.py").write_text("def foo(): pass")

    saver = InMemorySaver()
    graph = build_graph(checkpointer=saver)
    initial_state = {
        "mode": "learn",
        "messages": [],
        "current_speaker": "questioner",
        "hitl_flags": ["questioner"],
        "turn_count": 0,
        "max_turns": 1,
        "context_strategy": "direct",
        "source_path": str(tmp_path),
    }
    config = {"configurable": {"thread_id": "hitl-userinput-1"}}
    user_question = "foo 函数到底是干嘛的？"

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "It does nothing."

    mock_provider = MagicMock()
    mock_provider.retrieve.return_value = ["chunk1", "chunk2"]
    mock_provider.get_overview.return_value = ["overview"]

    from biteme.cli import _run_graph

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), patch(
        "biteme.graph.nodes.create_provider", return_value=mock_provider
    ), patch("biteme.cli.typer.prompt", return_value=user_question):
        _run_graph(graph, initial_state, config)

    final_state = graph.get_state(config).values
    assert final_state is not None

    speakers = [m["speaker"] for m in final_state["messages"]]
    contents = [m["content"] for m in final_state["messages"]]

    assert "human" in speakers, "human turn missing"
    assert user_question in contents, "user input not recorded as human turn"

    retrieve_args = [call.args[0] for call in mock_provider.retrieve.call_args_list]
    assert user_question in retrieve_args, (
        f"answerer never retrieved with user input; saw: {retrieve_args}"
    )

    assert any(m["speaker"] == "answerer" for m in final_state["messages"]), (
        "answerer never produced a turn"
    )
