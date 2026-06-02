import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from biteme.graph.state import SessionState, KeywordScore


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
        llm_reference_answer="",
        review_history=[],
    )
    defaults.update(kwargs)
    return defaults  # type: ignore


def _make_llm_mock(responses: list[str]) -> MagicMock:
    """Return a ChatOpenAI mock whose .invoke() cycles through `responses`."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content=r) for r in responses
    ]
    return mock_llm


# ---------------------------------------------------------------------------
# Test 1: empty review_history → no-op
# ---------------------------------------------------------------------------

def test_memory_node_skips_when_review_history_empty(tmp_path):
    from biteme.graph.nodes import memory_node

    state = make_state(review_history=[])
    memory_file = tmp_path / "review_memory.json"

    with patch("biteme.graph.nodes.settings") as mock_settings, \
         patch("biteme.graph.nodes.ChatOpenAI") as mock_chat:
        mock_settings.review_memory_path = memory_file
        mock_settings.openai_model = "gpt-4o"
        result = memory_node(state)

    assert result == {}
    assert not memory_file.exists()
    mock_chat.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: first run – no existing file → creates review_memory.json
# ---------------------------------------------------------------------------

def test_memory_node_first_run_creates_file(tmp_path):
    from biteme.graph.nodes import memory_node

    review_history = [
        [{"keyword": "向量检索", "score": 7}],
        [{"keyword": "梯度消失", "score": 4}],
    ]
    state = make_state(review_history=review_history)
    memory_file = tmp_path / "review_memory.json"

    llm_response = '[{"keyword": "向量检索", "scores": [7]}, {"keyword": "梯度消失", "scores": [4]}]'
    mock_llm = _make_llm_mock([llm_response])

    with patch("biteme.graph.nodes.settings") as mock_settings, \
         patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm):
        mock_settings.review_memory_path = memory_file
        mock_settings.openai_model = "gpt-4o"
        result = memory_node(state)

    assert result == {}
    assert memory_file.exists()
    data = json.loads(memory_file.read_text())
    assert isinstance(data, list)
    assert len(data) == 2
    kw_map = {e["keyword"]: e for e in data}
    assert kw_map["向量检索"]["scores"] == [7]
    assert kw_map["向量检索"]["avg_score"] == 7.0
    assert kw_map["梯度消失"]["scores"] == [4]
    assert kw_map["梯度消失"]["avg_score"] == 4.0


# ---------------------------------------------------------------------------
# Test 3: existing memory + new keywords → old entries preserved, new appended
# ---------------------------------------------------------------------------

def test_memory_node_appends_new_keywords(tmp_path):
    from biteme.graph.nodes import memory_node

    old_memory = [{"keyword": "旧关键词", "scores": [6], "avg_score": 6.0}]
    memory_file = tmp_path / "review_memory.json"
    memory_file.write_text(json.dumps(old_memory, ensure_ascii=False))

    review_history = [[{"keyword": "新关键词", "score": 9}]]
    state = make_state(review_history=review_history)

    llm_response = '[{"keyword": "旧关键词", "scores": [6]}, {"keyword": "新关键词", "scores": [9]}]'
    mock_llm = _make_llm_mock([llm_response])

    with patch("biteme.graph.nodes.settings") as mock_settings, \
         patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm):
        mock_settings.review_memory_path = memory_file
        mock_settings.openai_model = "gpt-4o"
        result = memory_node(state)

    assert result == {}
    data = json.loads(memory_file.read_text())
    kw_map = {e["keyword"]: e for e in data}
    assert kw_map["旧关键词"]["scores"] == [6]
    assert kw_map["旧关键词"]["avg_score"] == 6.0
    assert kw_map["新关键词"]["scores"] == [9]
    assert kw_map["新关键词"]["avg_score"] == 9.0


# ---------------------------------------------------------------------------
# Test 4: semantic merge – LLM combines synonyms, avg_score computed by code
# ---------------------------------------------------------------------------

def test_memory_node_semantic_merge_avg_score_by_code(tmp_path):
    from biteme.graph.nodes import memory_node

    old_memory = [{"keyword": "向量检索", "scores": [5], "avg_score": 5.0}]
    memory_file = tmp_path / "review_memory.json"
    memory_file.write_text(json.dumps(old_memory, ensure_ascii=False))

    # New session has "向量搜索" which LLM recognises as same as "向量检索"
    review_history = [[{"keyword": "向量搜索", "score": 8}]]
    state = make_state(review_history=review_history)

    # LLM merges them under canonical name "向量检索" with both scores
    llm_response = '[{"keyword": "向量检索", "scores": [5, 8]}]'
    mock_llm = _make_llm_mock([llm_response])

    with patch("biteme.graph.nodes.settings") as mock_settings, \
         patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm):
        mock_settings.review_memory_path = memory_file
        mock_settings.openai_model = "gpt-4o"
        result = memory_node(state)

    assert result == {}
    data = json.loads(memory_file.read_text())
    assert len(data) == 1
    entry = data[0]
    assert entry["keyword"] == "向量检索"
    assert entry["scores"] == [5, 8]
    # avg_score computed by code: round((5+8)/2, 2) = 6.5
    assert entry["avg_score"] == 6.5


# ---------------------------------------------------------------------------
# Test 5: invalid JSON on first attempt, valid on second → file written
# ---------------------------------------------------------------------------

def test_memory_node_retry_succeeds_on_second_attempt(tmp_path):
    from biteme.graph.nodes import memory_node

    review_history = [[{"keyword": "embedding", "score": 6}]]
    state = make_state(review_history=review_history)
    memory_file = tmp_path / "review_memory.json"

    bad_response = "这不是JSON"
    good_response = '[{"keyword": "embedding", "scores": [6]}]'
    mock_llm = _make_llm_mock([bad_response, good_response])

    with patch("biteme.graph.nodes.settings") as mock_settings, \
         patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm):
        mock_settings.review_memory_path = memory_file
        mock_settings.openai_model = "gpt-4o"
        result = memory_node(state)

    assert result == {}
    assert memory_file.exists()
    data = json.loads(memory_file.read_text())
    assert data[0]["keyword"] == "embedding"
    assert data[0]["avg_score"] == 6.0
    assert mock_llm.invoke.call_count == 2


# ---------------------------------------------------------------------------
# Test 6: all 3 retries fail → no file written, no exception raised
# ---------------------------------------------------------------------------

def test_memory_node_all_retries_fail_no_exception(tmp_path):
    from biteme.graph.nodes import memory_node

    review_history = [[{"keyword": "embedding", "score": 6}]]
    state = make_state(review_history=review_history)
    memory_file = tmp_path / "review_memory.json"

    mock_llm = _make_llm_mock(["bad1", "bad2", "bad3"])

    with patch("biteme.graph.nodes.settings") as mock_settings, \
         patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.Console"):
        mock_settings.review_memory_path = memory_file
        mock_settings.openai_model = "gpt-4o"
        result = memory_node(state)

    assert result == {}
    assert not memory_file.exists()
    assert mock_llm.invoke.call_count == 3


# ---------------------------------------------------------------------------
# Test 7: file write failure → no exception propagates
# ---------------------------------------------------------------------------

def test_memory_node_write_failure_no_exception(tmp_path):
    from biteme.graph.nodes import memory_node

    review_history = [[{"keyword": "RLHF", "score": 5}]]
    state = make_state(review_history=review_history)
    memory_file = tmp_path / "review_memory.json"

    llm_response = '[{"keyword": "RLHF", "scores": [5]}]'
    mock_llm = _make_llm_mock([llm_response])

    def bad_write(*args, **kwargs):
        raise OSError("disk full")

    with patch("biteme.graph.nodes.settings") as mock_settings, \
         patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.Console"), \
         patch("pathlib.Path.write_text", side_effect=bad_write):
        mock_settings.review_memory_path = memory_file
        mock_settings.openai_model = "gpt-4o"
        result = memory_node(state)  # must not raise

    assert result == {}


# ---------------------------------------------------------------------------
# Test 8: corrupt existing file → treated as empty, new data still written
# ---------------------------------------------------------------------------

def test_memory_node_corrupt_file_treated_as_empty(tmp_path):
    from biteme.graph.nodes import memory_node

    memory_file = tmp_path / "review_memory.json"
    memory_file.write_text("this is not json")

    review_history = [[{"keyword": "embedding", "score": 5}]]
    state = make_state(review_history=review_history)
    llm_response = '[{"keyword": "embedding", "scores": [5]}]'
    mock_llm = _make_llm_mock([llm_response])

    with patch("biteme.graph.nodes.settings") as mock_settings, \
         patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm):
        mock_settings.review_memory_path = memory_file
        mock_settings.openai_model = "gpt-4o"
        result = memory_node(state)

    assert result == {}
    assert memory_file.exists()
    data = json.loads(memory_file.read_text())
    assert any(e["keyword"] == "embedding" for e in data)
