import json
import os
import pytest
from datetime import date
from pathlib import Path
from biteme.graph.memory import (
    apply_updates,
    load_memory,
    MemoryUpdate,
    save_memory,
    _parse_memory_updates,
)


def test_parse_memory_updates_strips_markdown_fence():
    content = """```json
{
  "updates": [
    {
      "key": "llm_context_window",
      "aliases": [],
      "score": 7,
      "strength": "good",
      "weakness": "bad"
    }
  ]
}
```"""
    parsed = _parse_memory_updates(content)
    assert len(parsed.updates) == 1
    assert parsed.updates[0].key == "llm_context_window"
    assert parsed.updates[0].score == 7


def test_load_memory_missing_file(tmp_path):
    path = tmp_path / "memory.json"
    result = load_memory(path)
    assert result == {"entries": {}}


def test_load_memory_existing_file(tmp_path):
    path = tmp_path / "memory.json"
    data = {"entries": {"foo": {"aliases": ["bar"], "recent_scores": [7],
                                "avg_score": 7.0, "last_update": "2026-06-07",
                                "comments": {"strength": [], "weakness": []}}}}
    path.write_text(json.dumps(data))
    result = load_memory(path)
    assert result == data


def test_save_memory_roundtrip(tmp_path):
    path = tmp_path / "memory.json"
    data = {"entries": {"test_key": {"aliases": ["Test"], "recent_scores": [5],
                                     "avg_score": 5.0, "last_update": "2026-06-07",
                                     "comments": {"strength": ["ok"], "weakness": ["meh"]}}}}
    save_memory(data, path)
    assert load_memory(path) == data


def test_save_memory_atomic(tmp_path):
    """save_memory must not leave a tmp file behind on success."""
    path = tmp_path / "memory.json"
    save_memory({"entries": {}}, path)
    tmp_files = [f for f in tmp_path.iterdir() if f.name != "memory.json"]
    assert tmp_files == []


def _make_update(key, aliases, score, strength, weakness):
    return MemoryUpdate(key=key, aliases=aliases,
                        score=score, strength=strength, weakness=weakness)


def test_apply_updates_creates_new_entry():
    data = {"entries": {}}
    updates = [_make_update("python_generators",
                             ["Python 生成器", "yield"],
                             7, "good", "missing send()")]
    apply_updates(data, updates)
    entry = data["entries"]["python_generators"]
    assert entry["aliases"] == ["Python 生成器", "yield"]
    assert entry["recent_scores"] == [7]
    assert entry["avg_score"] == 7.0
    assert entry["comments"]["strength"] == ["good"]
    assert entry["comments"]["weakness"] == ["missing send()"]
    assert entry["last_update"] == date.today().isoformat()


def test_apply_updates_appends_to_existing():
    data = {"entries": {"python_generators": {
        "aliases": ["Python 生成器"],
        "recent_scores": [6],
        "avg_score": 6.0,
        "last_update": "2026-01-01",
        "comments": {"strength": ["ok"], "weakness": ["bad"]},
    }}}
    updates = [_make_update("python_generators", ["yield"], 8, "better", "still missing send()")]
    apply_updates(data, updates)
    entry = data["entries"]["python_generators"]
    assert entry["recent_scores"] == [6, 8]
    assert round(entry["avg_score"], 2) == 7.0
    assert entry["comments"]["strength"] == ["ok", "better"]
    assert entry["comments"]["weakness"] == ["bad", "still missing send()"]


def test_apply_updates_alias_dedup():
    data = {"entries": {"python_generators": {
        "aliases": ["Python 生成器", "yield"],
        "recent_scores": [6],
        "avg_score": 6.0,
        "last_update": "2026-01-01",
        "comments": {"strength": [], "weakness": []},
    }}}
    # "yield" already exists; "惰性求值" is new
    updates = [_make_update("python_generators", ["yield", "惰性求值"], 7, "s", "w")]
    apply_updates(data, updates)
    entry = data["entries"]["python_generators"]
    assert entry["aliases"].count("yield") == 1
    assert "惰性求值" in entry["aliases"]
    assert entry["aliases"] == ["Python 生成器", "yield", "惰性求值"]


def test_apply_updates_avg_score_multiple():
    data = {"entries": {}}
    updates = [_make_update("foo", ["Foo"], 4, "s", "w")]
    apply_updates(data, updates)
    updates2 = [_make_update("foo", [], 6, "s2", "w2")]
    apply_updates(data, updates2)
    assert data["entries"]["foo"]["avg_score"] == 5.0


# ---------------------------------------------------------------------------
# memory_node tests
# ---------------------------------------------------------------------------

from unittest.mock import patch
from biteme.graph.memory import memory_node


def _base_state():
    return {
        "hitl_flags": [],          # no HITL — should be a no-op
        "messages": [
            {"speaker": "questioner", "content": "What is a generator?", "retrieved_chunks": []},
            {"speaker": "human", "content": "It yields values.", "retrieved_chunks": []},
        ],
        "llm_reference_answer": "A generator is...",
        "mode": "learn",
        "source_path": "/tmp/fake",
    }


def test_memory_node_passthrough_when_not_hitl(tmp_path):
    state = _base_state()
    with patch("biteme.graph.memory.settings") as mock_settings:
        mock_settings.biteme_home = tmp_path
        result = memory_node(state)
    assert result == {}
    assert not (tmp_path / "memory.json").exists()


def test_load_memory_corrupt_file(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{invalid json{{")
    assert load_memory(path) == {"entries": {}}
