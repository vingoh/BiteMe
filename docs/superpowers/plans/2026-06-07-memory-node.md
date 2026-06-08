# Memory Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `memory_node` to the LangGraph that runs after each HITL answerer turn, scoring user knowledge mastery and persisting results to `~/.biteme/memory.json`.

**Architecture:** A new `biteme/graph/memory.py` module owns all memory logic (schemas, file I/O, LLM call, node function). The node is inserted linearly after `answerer` in the graph and is a no-op when not in HITL answerer mode. All state reads happen from existing fields; no new state fields are added.

**Tech Stack:** LangGraph, LangChain (`ChatOpenAI`, `with_structured_output`), Pydantic v2, Python stdlib (`json`, `os`, `tempfile`, `datetime`, `statistics`)

---

## File Map


| Action | Path                      | Responsibility                                                            |
| ------ | ------------------------- | ------------------------------------------------------------------------- |
| Create | `biteme/graph/memory.py`  | TypedDicts, Pydantic schemas, `load_memory`, `save_memory`, `memory_node` |
| Create | `tests/test_memory.py`    | Unit tests for all memory module functions                                |
| Modify | `biteme/graph/prompts.py` | Add `MEMORY_UPDATER` prompt, expose in `get_prompts()`                    |
| Modify | `biteme/graph/graph.py`   | Register `memory_node`, wire `answerer → memory` edge                     |


---

## Task 1: Memory file I/O — `load_memory` and `save_memory`

**Files:**

- Create: `biteme/graph/memory.py`
- Create: `tests/test_memory.py`
- **Step 1: Write failing tests for `load_memory` and `save_memory`**

Create `tests/test_memory.py`:

```python
import json
import os
import pytest
from pathlib import Path
from biteme.graph.memory import load_memory, save_memory


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
```

- **Step 2: Run tests to verify they fail**

```bash
conda run -n agent pytest tests/test_memory.py -v
```

Expected: `ImportError` — `biteme.graph.memory` does not exist yet.

- **Step 3: Implement `load_memory` and `save_memory` in `biteme/graph/memory.py`**

```python
from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from statistics import mean
from typing import TypedDict

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# TypedDicts for the in-memory representation of memory.json
# ---------------------------------------------------------------------------

class MemoryComments(TypedDict):
    strength: list[str]
    weakness: list[str]


class MemoryEntry(TypedDict):
    aliases: list[str]
    recent_scores: list[int]
    avg_score: float
    last_update: str          # YYYY-MM-DD
    comments: MemoryComments


class MemoryFile(TypedDict):
    entries: dict[str, MemoryEntry]


# ---------------------------------------------------------------------------
# Pydantic schemas for LLM structured output
# ---------------------------------------------------------------------------

class MemoryUpdate(BaseModel):
    key: str
    aliases: list[str]   # if key is new: initial aliases; if key exists: new aliases to add (may be empty)
    score: int                # 0-10
    strength: str | None      # specific strength; null if none identifiable
    weakness: str | None      # specific error/omission; null if none identifiable


class MemoryUpdates(BaseModel):
    updates: list[MemoryUpdate]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_memory(path: Path) -> MemoryFile:
    """Load memory.json; return empty structure if file is absent."""
    if not path.exists():
        return {"entries": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(data: MemoryFile, path: Path) -> None:
    """Atomically write memory data to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise
```

- **Step 4: Run tests to verify they pass**

```bash
conda run -n agent pytest tests/test_memory.py -v
```

Expected: all 4 tests PASS.

- **Step 5: Commit**

```bash
git add biteme/graph/memory.py tests/test_memory.py
git commit -m "feat: add memory file I/O (load_memory, save_memory)"
```

---

## Task 2: Write logic — `apply_updates`

**Files:**

- Modify: `biteme/graph/memory.py`
- Modify: `tests/test_memory.py`
- **Step 1: Write failing tests for `apply_updates`**

Append to `tests/test_memory.py`:

```python
from biteme.graph.memory import apply_updates, MemoryUpdate


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


def test_apply_updates_avg_score_multiple():
    data = {"entries": {}}
    updates = [_make_update("foo", ["Foo"], 4, "s", "w")]
    apply_updates(data, updates)
    updates2 = [_make_update("foo", [], 6, "s2", "w2")]
    apply_updates(data, updates2)
    assert data["entries"]["foo"]["avg_score"] == 5.0
```

- **Step 2: Run tests to verify they fail**

```bash
conda run -n agent pytest tests/test_memory.py::test_apply_updates_creates_new_entry -v
```

Expected: `ImportError` — `apply_updates` not defined yet.

- **Step 3: Implement `apply_updates` in `biteme/graph/memory.py`**

Add after `save_memory`:

```python
def apply_updates(data: MemoryFile, updates: list[MemoryUpdate]) -> None:
    """Apply LLM-produced updates to the in-memory data dict (mutates in place)."""
    today = date.today().isoformat()
    for update in updates:
        if update.key not in data["entries"]:
            data["entries"][update.key] = {
                "aliases": list(update.aliases),
                "recent_scores": [],
                "avg_score": 0.0,
                "last_update": today,
                "comments": {"strength": [], "weakness": []},
            }
        entry = data["entries"][update.key]

        # Merge aliases (order-preserving dedup)
        existing_set: set[str] = set(entry["aliases"])
        for alias in update.aliases:
            if alias not in existing_set:
                entry["aliases"].append(alias)
                existing_set.add(alias)

        entry["recent_scores"].append(update.score)
        entry["avg_score"] = mean(entry["recent_scores"])
        entry["last_update"] = today
        if update.strength is not None:
            entry["comments"]["strength"].append(update.strength)
        if update.weakness is not None:
            entry["comments"]["weakness"].append(update.weakness)
```

- **Step 4: Run tests to verify they pass**

```bash
conda run -n agent pytest tests/test_memory.py -v
```

Expected: all tests PASS.

- **Step 5: Commit**

```bash
git add biteme/graph/memory.py tests/test_memory.py
git commit -m "feat: add apply_updates logic with alias dedup and avg_score"
```

---

## Task 3: Memory prompt in `prompts.py`

**Files:**

- Modify: `biteme/graph/prompts.py`
- **Step 1: Add `MEMORY_UPDATER` prompt constant**

Open `biteme/graph/prompts.py` and append the following constant before `get_prompts()`:

```python
MEMORY_UPDATER = """\
你是一个知识点追踪助手。根据提供的一问一答，识别其中涉及的 1–3 个知识点，
并评估用户对每个知识点的掌握程度。

## 已有知识点
{existing_keys}

每个已有知识点包含：
- key：稳定标识符
- aliases：该知识点的别名或同义说法

## 本轮问答
问题：{question}
用户回答：{user_answer}
LLM 参考答案：{llm_reference}

## 任务
1. 根据“问题”和“LLM 参考答案”，识别本轮真正考察的 1–3 个核心知识点。
2. 对每个知识点，先尝试匹配“已有知识点”：
   - 如果当前知识点与某个已有 key / aliases 表示的是同一个可学习概念，必须复用已有 key。
   - 如果只是话题相关、上下游相关、或者名称相似但考察重点不同，不要复用。
   - 如果没有合适的已有 key，创建新的 snake_case key。
3. 根据“用户回答”相对于“LLM 参考答案”的准确性、完整性和深度，对每个知识点打 0–10 分。
4. 给出一句 strength 和一句 weakness，必须具体对应用户本轮回答，不要泛泛而谈。

## 输出规则
只输出合法 JSON，不要 Markdown 代码块，不要解释文字。

输出格式：
{
  "items": [
    {
      "key": "snake_case_key",
      "aliases": ["本轮出现的新别名或同义说法"],
      "score": 0,
      "strength": "一句话说明用户在该知识点上的优点",
      "weakness": "一句话说明用户在该知识点上的不足或遗漏"
    }
  ]
}

## 约束
- items 数量为 1–3。
- key 必须是英文小写 snake_case。
- 如果复用已有 key，key 必须与已有知识点中的 key 完全一致。
- aliases 只放本轮新出现、且有助于后续匹配的别名；没有则为空数组。
- score 必须是 0–10 的整数。
- strength：只有当用户回答中存在具体、可定位的优点时才输出；否则为 null。禁止泛泛评价，例如"回答较完整""表达清晰"。
- weakness：只有当用户回答中存在具体错误、遗漏或表达不清时才输出；否则为 null。禁止泛泛评价，例如"理解不够深入""还需加强"。
- 不要输出过宽泛的 key，例如 mechanism_understanding、design_thinking、basic_concept。
"""
```

- **Step 2: Expose prompt in `get_prompts()`**

In `get_prompts()`, add `"memory": MEMORY_UPDATER` to both return dicts:

```python
def get_prompts(mode: str) -> dict[str, str]:
    if mode == "learn":
        return {
            "questioner": LEARN_QUESTIONER,
            "answerer": LEARN_ANSWERER,
            "planner": LEARN_PLANNER,
            "memory": MEMORY_UPDATER,
        }
    return {
        "questioner": INTERVIEW_QUESTIONER,
        "answerer": INTERVIEW_ANSWERER,
        "planner": INTERVIEW_PLANNER,
        "memory": MEMORY_UPDATER,
    }
```

- **Step 3: Verify no import errors**

```bash
conda run -n agent python -c "from biteme.graph.prompts import get_prompts; p = get_prompts('learn'); assert 'memory' in p; print('OK')"
```

Expected output: `OK`

- **Step 4: Commit**

```bash
git add biteme/graph/prompts.py
git commit -m "feat: add MEMORY_UPDATER prompt"
```

---

## Task 4: `memory_node` function

**Files:**

- Modify: `biteme/graph/memory.py`
- Modify: `tests/test_memory.py`
- **Step 1: Write failing test for non-HITL pass-through**

Append to `tests/test_memory.py`:

```python
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
```

- **Step 2: Run test to verify it fails**

```bash
conda run -n agent pytest tests/test_memory.py::test_memory_node_passthrough_when_not_hitl -v
```

Expected: `ImportError` — `memory_node` not defined yet.

- **Step 3: Implement `memory_node` in `biteme/graph/memory.py`**

Add the following imports at the top of the file (after existing imports):

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..config import settings
from .prompts import get_prompts
from .state import SessionState
```

Then add `memory_node` after `apply_updates`:

```python
def memory_node(state: SessionState) -> dict:
    if "answerer" not in state["hitl_flags"]:
        return {}

    messages = state["messages"]
    if len(messages) < 2:
        return {}

    question = messages[-2]["content"]
    user_answer = messages[-1]["content"]
    llm_reference = state.get("llm_reference_answer", "")

    memory_path = settings.biteme_home / "memory.json"
    data = load_memory(memory_path)

    existing_keys = [
        {"key": k, "aliases": v["aliases"]}
        for k, v in data["entries"].items()
    ]

    prompts = get_prompts(state["mode"])
    prompt_text = prompts["memory"].format(
        existing_keys=existing_keys,
        question=question,
        user_answer=user_answer,
        llm_reference=llm_reference,
    )

    llm = ChatOpenAI(model=settings.openai_model, temperature=0.2)
    structured_llm = llm.with_structured_output(MemoryUpdates)
    result: MemoryUpdates = structured_llm.invoke(
        [HumanMessage(content=prompt_text)]
    )

    apply_updates(data, result.updates)
    save_memory(data, memory_path)

    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    summary = "\n".join(
        f"  [{u.key}] score={u.score}  strength: {u.strength}  weakness: {u.weakness}"
        for u in result.updates
    )
    console.print(Panel(summary, title="[magenta]Memory Updated[/magenta]"))

    return {}
```

- **Step 4: Run pass-through test to verify it passes**

```bash
conda run -n agent pytest tests/test_memory.py::test_memory_node_passthrough_when_not_hitl -v
```

Expected: PASS.

- **Step 5: Run full test suite**

```bash
conda run -n agent pytest tests/test_memory.py -v
```

Expected: all tests PASS.

- **Step 6: Commit**

```bash
git add biteme/graph/memory.py tests/test_memory.py
git commit -m "feat: implement memory_node with LLM structured output"
```

---

## Task 5: Wire `memory_node` into the graph

**Files:**

- Modify: `biteme/graph/graph.py`
- **Step 1: Update `graph.py`**

Replace the full contents of `biteme/graph/graph.py` with:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import SessionState
from .nodes import planner_node, questioner_node, answerer_node
from .memory import memory_node


def _should_continue(state: SessionState) -> str:
    if state["turn_count"] >= state["max_turns"]:
        return END
    return "continue"


def build_graph(checkpointer: SqliteSaver) -> StateGraph:
    builder = StateGraph(SessionState)
    builder.add_node("planner", planner_node)
    builder.add_node("questioner", questioner_node)
    builder.add_node("answerer", answerer_node)
    builder.add_node("memory", memory_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "questioner")
    builder.add_edge("questioner", "answerer")
    builder.add_edge("answerer", "memory")
    builder.add_conditional_edges(
        "memory",
        _should_continue,
        {"continue": "questioner", END: END},
    )
    return builder.compile(checkpointer=checkpointer, interrupt_before=[])
```

- **Step 2: Verify graph builds without error**

```bash
conda run -n agent python -c "
from langgraph.checkpoint.sqlite import SqliteSaver
from biteme.graph.graph import build_graph
with SqliteSaver.from_conn_string(':memory:') as cp:
    g = build_graph(cp)
    print('nodes:', list(g.nodes))
"
```

Expected output includes: `nodes: ['planner', 'questioner', 'answerer', 'memory', ...]`

- **Step 3: Run full test suite**

```bash
conda run -n agent pytest tests/ -v --ignore=tests/test_hitl_integration.py
```

Expected: all tests PASS (excluding integration test which requires live API keys).

- **Step 4: Commit**

```bash
git add biteme/graph/graph.py
git commit -m "feat: wire memory_node into graph after answerer"
```

---

## Self-Review

**Spec coverage check:**


| Spec requirement                                  | Task                                                                        |
| ------------------------------------------------- | --------------------------------------------------------------------------- |
| memory_node runs after HITL answerer turn         | Task 4 + Task 5                                                             |
| Pass-through when not HITL                        | Task 4 (pass-through test)                                                  |
| Inputs: question, user answer, LLM reference      | Task 4 (memory_node reads messages[-2], messages[-1], llm_reference_answer) |
| LLM outputs 1-3 keys with score/strength/weakness | Task 3 (prompt) + Task 4 (MemoryUpdates schema)                             |
| Match definition: ≥80% overlap required           | Task 3 (MEMORY_UPDATER prompt)                                              |
| Existing key reuse via aliases                    | Task 3 (prompt) + Task 2 (apply_updates)                                    |
| New key creation                                  | Task 2 (apply_updates is_new=True branch)                                   |
| aliases append with order-preserving dedup        | Task 2 (test_apply_updates_alias_dedup)                                     |
| recent_scores append                              | Task 2                                                                      |
| avg_score computed in Python                      | Task 2                                                                      |
| last_update as YYYY-MM-DD                         | Task 2 (date.today().isoformat())                                           |
| comments.strength / weakness as lists             | Task 2                                                                      |
| Atomic write                                      | Task 1 (save_memory uses tempfile + os.replace)                             |
| Global memory file at ~/.biteme/memory.json       | Task 4 (settings.biteme_home / "memory.json")                               |
| Graph edge: answerer → memory → _should_continue  | Task 5                                                                      |


