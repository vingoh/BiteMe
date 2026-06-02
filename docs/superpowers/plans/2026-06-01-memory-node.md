# Memory Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `memory_node` that runs at session end, merges the session's `review_history` into `~/.biteme/review_memory.json` via a single LLM call, and persists per-keyword mastery scores for cross-session use.

**Architecture:** A new side-effect graph node reads the persistent memory file, calls the LLM once to merge and deduplicate semantically similar keywords from the current `review_history`, then code computes `avg_score` and writes the result back. All terminal graph paths are redirected through this node before `END`.

**Tech Stack:** Python 3.10, LangGraph, LangChain (`ChatOpenAI`, `langchain_core.messages`), `unittest.mock` for tests.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `biteme/config.py` | Modify | Add `review_memory_path` attribute |
| `biteme/graph/prompts.py` | Modify | Add `MEMORY_MERGE` prompt constant |
| `biteme/graph/nodes.py` | Modify | Add `memory_node()` function |
| `biteme/graph/graph.py` | Modify | Wire `memory_node`; redirect `END` branches |
| `tests/test_memory_node.py` | Create | Unit tests for `memory_node` |
| `tests/test_graph_nodes.py` | Modify | Update routing assertions that expect `END` |

---

## Task 1: Add `review_memory_path` to Settings

**Files:**
- Modify: `biteme/config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_settings_has_review_memory_path():
    from biteme.config import settings
    from pathlib import Path
    assert hasattr(settings, "review_memory_path")
    assert isinstance(settings.review_memory_path, Path)
    assert settings.review_memory_path.name == "review_memory.json"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n agent pytest tests/test_config.py::test_settings_has_review_memory_path -v
```

Expected: FAIL with `AttributeError` or assertion error.

- [ ] **Step 3: Add `review_memory_path` to `Settings`**

In `biteme/config.py`, add the attribute inside `__init__` right after the existing `sessions_dir` line:

```python
self.review_memory_path: Path = self.biteme_home / "review_memory.json"
```

No directory to create — it's a file path, not a dir. `ensure_dirs()` is unchanged.

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n agent pytest tests/test_config.py::test_settings_has_review_memory_path -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biteme/config.py tests/test_config.py
git commit -m "feat: add review_memory_path to Settings"
```

---

## Task 2: Add `MEMORY_MERGE` prompt

**Files:**
- Modify: `biteme/graph/prompts.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graph_nodes.py`:

```python
def test_memory_merge_prompt_exists():
    from biteme.graph.prompts import MEMORY_MERGE
    assert isinstance(MEMORY_MERGE, str)
    assert "review_history" in MEMORY_MERGE
    assert "scores" in MEMORY_MERGE
    assert "avg_score" not in MEMORY_MERGE  # avg_score computed by code, not LLM
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_memory_merge_prompt_exists -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add `MEMORY_MERGE` to `prompts.py`**

Append to `biteme/graph/prompts.py` before the `get_prompts` function:

```python
MEMORY_MERGE = """\
你是用户学习记忆的管理员。你将收到：
1. 旧记忆（JSON 数组，可能为空）
2. 本次 session 的 review_history（每轮关键词及整数得分）

任务：
- 将 review_history 中的关键词与旧记忆中的关键词进行语义比对，含义相近的视为同一条目，选定一个规范名称
- 同一条目：将本次所有相关轮次的得分（整数）**逐条**追加到旧 scores 列表末尾
- 新出现的关键词：创建新条目，scores 为本次各轮得分列表
- 旧记忆中未被本次覆盖的条目：原样保留（只保留 keyword 和 scores 字段）
- 不要计算或输出 avg_score，由调用方计算

只输出合法 JSON 数组，不含 Markdown 代码围栏或任何其他文字。
格式：[{"keyword": "...", "scores": [...]}]
"""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_memory_merge_prompt_exists -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biteme/graph/prompts.py tests/test_graph_nodes.py
git commit -m "feat: add MEMORY_MERGE prompt"
```

---

## Task 3: Write `memory_node` tests (all failing)

**Files:**
- Create: `tests/test_memory_node.py`

- [ ] **Step 1: Create the test file with all 7 test cases**

```python
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
```

- [ ] **Step 2: Run all tests to verify they all fail**

```bash
conda run -n agent pytest tests/test_memory_node.py -v
```

Expected: all 7 FAIL with `ImportError` (memory_node not yet defined).

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_memory_node.py
git commit -m "test: add failing tests for memory_node (TDD)"
```

---

## Task 4: Implement `memory_node`

**Files:**
- Modify: `biteme/graph/nodes.py`

- [ ] **Step 1: Add the import for `AIMessage` at the top of `nodes.py`**

The existing import line is:
```python
from langchain_core.messages import HumanMessage, SystemMessage
```

Change it to:
```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
```

- [ ] **Step 2: Add the `MEMORY_MERGE` import and `_MAX_MEMORY_RETRIES` constant**

Add below the existing imports in `nodes.py` (after the `from .prompts import get_prompts` line):

```python
from .prompts import MEMORY_MERGE

_MAX_MEMORY_RETRIES = 3
```

- [ ] **Step 3: Add `memory_node` function to `nodes.py`**

Append at the end of `biteme/graph/nodes.py`:

```python
def memory_node(state: SessionState) -> dict:
    if not state["review_history"]:
        return {}

    memory_path = settings.review_memory_path
    old_memory: list = []
    if memory_path.exists():
        try:
            old_memory = json.loads(memory_path.read_text(encoding="utf-8"))
        except Exception:
            Console().print("[yellow]警告：无法读取 review_memory.json，视为空记忆[/yellow]")
            old_memory = []

    old_memory_text = json.dumps(old_memory, ensure_ascii=False)
    review_history_text = json.dumps(state["review_history"], ensure_ascii=False)

    llm = ChatOpenAI(model=settings.openai_model, temperature=0.0)
    messages = [
        SystemMessage(content=MEMORY_MERGE),
        HumanMessage(content=(
            f"旧记忆：\n{old_memory_text}\n\n"
            f"本次 session 的 review_history：\n{review_history_text}"
        )),
    ]

    validated: list | None = None
    for attempt in range(_MAX_MEMORY_RETRIES):
        response = llm.invoke(messages)
        raw: str = response.content
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("顶层结构必须是 JSON 数组")
            entries = []
            for entry in data:
                kw = entry.get("keyword")
                scores = entry.get("scores")
                if (
                    isinstance(kw, str) and kw
                    and isinstance(scores, list)
                    and scores
                    and all(isinstance(s, int) and 0 <= s <= 10 for s in scores)
                ):
                    entries.append({"keyword": kw, "scores": scores})
            validated = entries
            break
        except Exception as exc:
            if attempt < _MAX_MEMORY_RETRIES - 1:
                messages.append(AIMessage(content=raw))
                messages.append(HumanMessage(content=(
                    f"上面的输出有误：{exc}。"
                    f"请修正并重新输出合法 JSON 数组，"
                    f'格式：[{{"keyword": "...", "scores": [...]}}]'
                )))
            else:
                Console().print("[yellow]警告：memory_node 连续 3 次输出无效，跳过写入[/yellow]")
                return {}

    result = [
        {
            "keyword": entry["keyword"],
            "scores": entry["scores"],
            "avg_score": round(sum(entry["scores"]) / len(entry["scores"]), 2),
        }
        for entry in (validated or [])
    ]

    try:
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        Console().print("[yellow]警告：review_memory.json 写入失败[/yellow]")

    return {}
```

- [ ] **Step 4: Run `test_memory_node.py` to verify all tests pass**

```bash
conda run -n agent pytest tests/test_memory_node.py -v
```

Expected: all 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add biteme/graph/nodes.py biteme/graph/prompts.py
git commit -m "feat: implement memory_node with LLM merge and retry logic"
```

---

## Task 5: Wire `memory_node` into the graph

**Files:**
- Modify: `biteme/graph/graph.py`
- Modify: `tests/test_graph_nodes.py`

- [ ] **Step 1: Update the two routing-assertion tests that currently expect `END`**

In `tests/test_graph_nodes.py`, find and update these two tests:

```python
# OLD:
def test_after_answerer_routes_to_end_when_max_turns_non_hitl():
    state = make_state(hitl_flags=[], turn_count=5, max_turns=5)
    from langgraph.graph import END
    assert _after_answerer(state) == END

# NEW:
def test_after_answerer_routes_to_memory_when_max_turns_non_hitl():
    state = make_state(hitl_flags=[], turn_count=5, max_turns=5)
    assert _after_answerer(state) == "memory"
```

```python
# OLD (inside the same file, _should_continue tested indirectly via graph):
# There is no direct test for _should_continue returning END; it's exercised
# through the graph topology test below. No change needed here.
```

Also update `test_graph_has_reviewer_node` to also assert `memory` exists:

```python
def test_graph_has_memory_node():
    from biteme.graph.graph import build_graph
    graph = build_graph(checkpointer=None)
    assert "memory" in set(graph.nodes.keys())
```

- [ ] **Step 2: Run updated routing test to verify it fails**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_after_answerer_routes_to_memory_when_max_turns_non_hitl tests/test_graph_nodes.py::test_graph_has_memory_node -v
```

Expected: FAIL (old graph still routes to `END`).

- [ ] **Step 3: Update `graph.py`**

Replace the contents of `biteme/graph/graph.py` with:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import SessionState
from .nodes import planner_node, questioner_node, answerer_node, reviewer_node, memory_node


def _after_answerer(state: SessionState) -> str:
    if "answerer" in state["hitl_flags"]:
        return "reviewer"
    if state["turn_count"] >= state["max_turns"]:
        return "memory"
    return "questioner"


def _should_continue(state: SessionState) -> str:
    if state["turn_count"] >= state["max_turns"]:
        return "memory"
    return "questioner"


def build_graph(checkpointer: SqliteSaver) -> StateGraph:
    builder = StateGraph(SessionState)
    builder.add_node("planner", planner_node)
    builder.add_node("questioner", questioner_node)
    builder.add_node("answerer", answerer_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("memory", memory_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "questioner")
    builder.add_edge("questioner", "answerer")
    builder.add_conditional_edges(
        "answerer",
        _after_answerer,
        {"reviewer": "reviewer", "questioner": "questioner", "memory": "memory"},
    )
    builder.add_conditional_edges(
        "reviewer",
        _should_continue,
        {"questioner": "questioner", "memory": "memory"},
    )
    builder.add_edge("memory", END)
    return builder.compile(checkpointer=checkpointer, interrupt_before=[])
```

- [ ] **Step 4: Run the full test suite**

```bash
conda run -n agent pytest tests/ -v
```

Expected: all tests PASS. Pay attention to:
- `test_after_answerer_routes_to_end_when_max_turns_non_hitl` — this test no longer exists (renamed in Step 1); if it still appears, delete the old version.
- `test_graph_has_memory_node` — PASS.
- All 7 `test_memory_node.py` tests — PASS.

- [ ] **Step 5: Commit**

```bash
git add biteme/graph/graph.py tests/test_graph_nodes.py
git commit -m "feat: wire memory_node into graph, redirect END branches"
```

---

## Self-Review

**Spec coverage:**
- [x] Storage format `review_memory.json` flat JSON array → Task 3 & 4
- [x] `review_memory_path` in `Settings` → Task 1
- [x] `MEMORY_MERGE` prompt, LLM temp 0.0, no `avg_score` in prompt → Task 2 & 4
- [x] Guard: empty `review_history` → no-op → Test 1
- [x] Read old memory, treat missing as `[]` → Task 4 Step 3
- [x] Single LLM call with retry up to 3 → Task 4 Step 3
- [x] Feed bad output + error back to LLM for retry → Task 4 Step 3
- [x] `avg_score` computed by code → Task 4 Step 3, Test 4
- [x] Write file, catch write errors → Task 4 Step 3, Test 7
- [x] Graph wiring: `END` → `memory` → `END` → Task 5
- [x] All 7 test scenarios → Task 3

**Placeholder scan:** No TBD/TODO found.

**Type consistency:**
- `memory_node` is imported in `graph.py` from `nodes.py` — consistent.
- `MEMORY_MERGE` is imported in `nodes.py` from `prompts.py` — consistent.
- `settings.review_memory_path` used in `memory_node` matches what's defined in `config.py` — consistent.
