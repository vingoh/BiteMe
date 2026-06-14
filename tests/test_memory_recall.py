import pytest
from unittest.mock import patch, MagicMock

from biteme.graph.memory import (
    RecalledEntry,
    MemoryRecallResult,
    _format_memory_entries_for_recall,
    _format_recalled_entries_for_refine,
    recall_memory,
    refine_question,
)
from biteme.graph.nodes import questioner_node
from biteme.graph.state import SessionState


SAMPLE_MEMORY = {
    "entries": {
        "bpe_tokenization": {
            "aliases": ["BPE", "Byte Pair Encoding", "子词分词"],
            "recent_scores": [3, 4],
            "avg_score": 3.5,
            "last_update": "2026-05-20",
            "comments": {
                "strength": ["用户能说出 BPE 的基本流程"],
                "weakness": ["用户未解释 OOV 处理方式", "未提到 vocabulary size 的影响"],
            },
        },
        "gradient_clipping": {
            "aliases": ["梯度裁剪", "clip_grad_norm"],
            "recent_scores": [8, 9],
            "avg_score": 8.5,
            "last_update": "2026-06-10",
            "comments": {
                "strength": ["正确描述了 clip_grad_norm 的用法"],
                "weakness": [],
            },
        },
    }
}


def test_format_memory_entries_includes_key_and_aliases():
    result = _format_memory_entries_for_recall(SAMPLE_MEMORY)
    assert "bpe_tokenization" in result
    assert "BPE" in result
    assert "Byte Pair Encoding" in result


def test_format_memory_entries_includes_recent_comments_only():
    # weakness 有 2 条，全部应出现（≤3）
    result = _format_memory_entries_for_recall(SAMPLE_MEMORY)
    assert "未解释 OOV 处理方式" in result
    assert "未提到 vocabulary size 的影响" in result


def test_format_memory_entries_empty():
    result = _format_memory_entries_for_recall({"entries": {}})
    assert result == ""


def test_format_recalled_entries_includes_score_and_date():
    recalled = [RecalledEntry(key="bpe_tokenization", relevance_reason='aliases 含 "BPE"')]
    result = _format_recalled_entries_for_refine(recalled, SAMPLE_MEMORY)
    assert "avg_score: 3.5" in result
    assert "last_update: 2026-05-20" in result
    assert 'aliases 含 "BPE"' in result


def test_format_recalled_entries_skips_missing_key():
    recalled = [RecalledEntry(key="nonexistent_key", relevance_reason="test")]
    result = _format_recalled_entries_for_refine(recalled, SAMPLE_MEMORY)
    assert result == ""


def test_format_recalled_entries_includes_weakness():
    recalled = [RecalledEntry(key="bpe_tokenization", relevance_reason="test")]
    result = _format_recalled_entries_for_refine(recalled, SAMPLE_MEMORY)
    assert "未解释 OOV 处理方式" in result


def test_recall_memory_empty_memory_returns_empty():
    result = recall_memory("BPE tokenization 是如何处理 OOV 词的？", {"entries": {}})
    assert result == []


def test_recall_memory_empty_draft_returns_empty():
    result = recall_memory("", SAMPLE_MEMORY)
    assert result == []


def test_recall_memory_returns_filtered_entries():
    mock_result = MemoryRecallResult(recalled=[
        RecalledEntry(key="bpe_tokenization", relevance_reason='aliases 含 "BPE"'),
        RecalledEntry(key="nonexistent_key", relevance_reason="test"),  # should be filtered
    ])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_result

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.with_structured_output.return_value = mock_llm
        result = recall_memory("BPE tokenization 是如何处理 OOV 词的？", SAMPLE_MEMORY)

    assert len(result) == 1
    assert result[0].key == "bpe_tokenization"


def test_recall_memory_llm_failure_returns_empty():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM error")

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.with_structured_output.return_value = mock_llm
        result = recall_memory("some question", SAMPLE_MEMORY)

    assert result == []


def test_recall_memory_returns_at_most_3():
    # LLM 返回 3 条都在 memory 中
    three_entries = [
        RecalledEntry(key="bpe_tokenization", relevance_reason="test"),
        RecalledEntry(key="gradient_clipping", relevance_reason="test"),
        RecalledEntry(key="bpe_tokenization", relevance_reason="duplicate"),  # key重复但还是过滤后≤3
    ]
    mock_result = MemoryRecallResult(recalled=three_entries)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_result

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.with_structured_output.return_value = mock_llm
        result = recall_memory("some question", SAMPLE_MEMORY)

    assert len(result) <= 3


SAMPLE_RECALLED = [
    RecalledEntry(key="bpe_tokenization", relevance_reason='aliases 含 "BPE"'),
]


def test_refine_question_returns_llm_output():
    mock_response = MagicMock()
    mock_response.content = "BPE 在遇到 OOV 词时会如何进行子词切分？请从 vocabulary 构建过程解释。"

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.invoke.return_value = mock_response
        result = refine_question("BPE 如何工作？", SAMPLE_RECALLED, SAMPLE_MEMORY)

    assert result == "BPE 在遇到 OOV 词时会如何进行子词切分？请从 vocabulary 构建过程解释。"


def test_refine_question_returns_draft_on_llm_failure():
    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.invoke.side_effect = Exception("LLM error")
        result = refine_question("BPE 如何工作？", SAMPLE_RECALLED, SAMPLE_MEMORY)

    assert result == "BPE 如何工作？"


def test_refine_question_returns_draft_on_empty_response():
    mock_response = MagicMock()
    mock_response.content = "   "  # blank

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.invoke.return_value = mock_response
        result = refine_question("BPE 如何工作？", SAMPLE_RECALLED, SAMPLE_MEMORY)

    assert result == "BPE 如何工作？"


def test_refine_question_uses_comments_in_prompt():
    """Verify the prompt sent to LLM includes comments content."""
    mock_response = MagicMock()
    mock_response.content = "refined question"
    captured_prompt = []

    def capture_invoke(messages):
        captured_prompt.append(messages[0].content)
        return mock_response

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.invoke.side_effect = capture_invoke
        refine_question("BPE 如何工作？", SAMPLE_RECALLED, SAMPLE_MEMORY)

    assert len(captured_prompt) == 1
    assert "未解释 OOV 处理方式" in captured_prompt[0]  # weakness 出现在 prompt 中
    assert "用户能说出 BPE 的基本流程" in captured_prompt[0]  # strength 出现在 prompt 中


def _make_state(**overrides) -> SessionState:
    base: SessionState = {
        "mode": "interview",
        "messages": [],
        "current_speaker": "questioner",
        "hitl_flags": [],
        "turn_count": 0,
        "max_turns": 5,
        "context_strategy": "direct",
        "source_path": "/tmp/fake",
        "outline": ["第一个话题"],
        "llm_reference_answer": "",
    }
    base.update(overrides)
    return base


def test_questioner_node_llm_path_calls_recall_and_refine(tmp_path):
    """When memory has entries, questioner should call recall_memory and refine_question."""
    state = _make_state()
    draft = "BPE 如何处理 OOV？"
    refined = "BPE 在遇到未知词时的子词切分机制是什么？"
    recalled = [RecalledEntry(key="bpe_tokenization", relevance_reason="test")]

    with (
        patch("biteme.graph.nodes.stream_agent", return_value=draft),
        patch("biteme.graph.nodes.load_memory", return_value=SAMPLE_MEMORY),
        patch("biteme.graph.nodes.recall_memory", return_value=recalled) as mock_recall,
        patch("biteme.graph.nodes.refine_question", return_value=refined) as mock_refine,
        patch("biteme.graph.nodes.create_provider") as mock_provider,
        patch("biteme.graph.nodes.create_agent"),
    ):
        mock_provider.return_value.get_overview.return_value = ["context"]
        result = questioner_node(state)

    mock_recall.assert_called_once_with(draft, SAMPLE_MEMORY)
    mock_refine.assert_called_once_with(draft, recalled, SAMPLE_MEMORY)
    assert result["messages"][-1]["content"] == refined


def test_questioner_node_llm_path_uses_draft_when_no_recall(tmp_path):
    """When recall returns empty, questioner uses draft directly."""
    state = _make_state()
    draft = "BPE 如何处理 OOV？"

    with (
        patch("biteme.graph.nodes.stream_agent", return_value=draft),
        patch("biteme.graph.nodes.load_memory", return_value={"entries": {}}),
        patch("biteme.graph.nodes.recall_memory", return_value=[]) as mock_recall,
        patch("biteme.graph.nodes.refine_question") as mock_refine,
        patch("biteme.graph.nodes.create_provider") as mock_provider,
        patch("biteme.graph.nodes.create_agent"),
    ):
        mock_provider.return_value.get_overview.return_value = ["context"]
        result = questioner_node(state)

    mock_recall.assert_called_once()
    mock_refine.assert_not_called()
    assert result["messages"][-1]["content"] == draft


def test_questioner_node_hitl_shows_recall_panel():
    """HITL path: when memory has entries, a panel with recalled memories is printed."""
    state = _make_state(hitl_flags=["questioner"])
    recalled = [RecalledEntry(key="bpe_tokenization", relevance_reason='aliases 含 "BPE"')]

    printed_panels = []

    class FakeConsole:
        def print(self, renderable, **kwargs):
            printed_panels.append(renderable)

    with (
        patch("biteme.graph.nodes.load_memory", return_value=SAMPLE_MEMORY),
        patch("biteme.graph.nodes.recall_memory", return_value=recalled),
        patch("biteme.graph.nodes.Console", return_value=FakeConsole()),
        patch("biteme.graph.nodes.interrupt", return_value="my question"),
    ):
        result = questioner_node(state)

    assert result["messages"][-1]["content"] == "my question"
    # Panel objects store content in .renderable; plain strings compare directly
    assert any(
        "bpe_tokenization" in str(getattr(p, "renderable", p))
        for p in printed_panels
    )


def test_questioner_node_hitl_silent_when_no_recall():
    """HITL path: when recalled is empty, no memory panel is shown."""
    state = _make_state(hitl_flags=["questioner"])
    printed_panels = []

    class FakeConsole:
        def print(self, renderable, **kwargs):
            printed_panels.append(str(renderable))

    with (
        patch("biteme.graph.nodes.load_memory", return_value={"entries": {}}),
        patch("biteme.graph.nodes.recall_memory", return_value=[]),
        patch("biteme.graph.nodes.Console", return_value=FakeConsole()),
        patch("biteme.graph.nodes.interrupt", return_value="my question"),
    ):
        result = questioner_node(state)

    assert not any("相关记忆参考" in t for t in printed_panels)
