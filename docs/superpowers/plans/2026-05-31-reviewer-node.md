# Reviewer Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `reviewer_node` that, after a user answers in HITL mode, compares the user's answer against the LLM reference answer, extracts 1–3 keywords from the question, and scores each keyword (0–10 integer) to surface weak/strong areas.

**Architecture:** Independent LangGraph node inserted between `answerer` and `questioner` in the HITL path. Graph routing splits after `answerer`: HITL goes to `reviewer`, non-HITL goes to `questioner`/END. `reviewer` uses a plain `ChatOpenAI.invoke()` call (no ReAct agent) returning structured JSON.

**Tech Stack:** LangGraph, LangChain (`ChatOpenAI`), `rich` for display, `json` stdlib for parsing.

---

## File Map

| File | Change |
|---|---|
| `biteme/graph/state.py` | Add `KeywordScore` TypedDict; add `review_history` field to `SessionState` |
| `biteme/graph/prompts.py` | Add `REVIEWER` constant; add `"reviewer"` key to `get_prompts()` |
| `biteme/graph/nodes.py` | Add `reviewer_node` function; add `json` import |
| `biteme/graph/graph.py` | Add `reviewer` node; replace `_should_continue` on `answerer` with `_after_answerer`; keep `_should_continue` on `reviewer` |
| `tests/test_graph_nodes.py` | Add `review_history` to `make_state()`; add `llm_reference_answer` to `make_state()`; add tests for `reviewer_node` and routing functions |

---

## Task 1: Update state.py

**Files:**
- Modify: `biteme/graph/state.py`
- Test: `tests/test_graph_nodes.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_graph_nodes.py
from biteme.graph.state import SessionState, Turn, KeywordScore

def test_keyword_score_typeddict():
    ks: KeywordScore = {"keyword": "向量检索", "score": 8}
    assert ks["keyword"] == "向量检索"
    assert ks["score"] == 8

def test_session_state_has_review_history_field():
    state = make_state()
    assert state["review_history"] == []

def test_session_state_review_history_accumulates():
    turn_keywords = [{"keyword": "embedding", "score": 9}]
    state = make_state(review_history=[turn_keywords])
    assert state["review_history"] == [turn_keywords]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_keyword_score_typeddict tests/test_graph_nodes.py::test_session_state_has_review_history_field -v
```

Expected: `ImportError: cannot import name 'KeywordScore'`

- [ ] **Step 3: Update state.py**

Replace the full content of `biteme/graph/state.py` with:

```python
from typing import Literal
from typing_extensions import TypedDict


class Turn(TypedDict):
    speaker: str            # "questioner" | "answerer" | "human"
    content: str
    retrieved_chunks: list[str]  # answerer 轮携带，其余为空列表


class KeywordScore(TypedDict):
    keyword: str   # e.g. "向量检索"
    score: int     # 0–10 integer; how well the user covered this concept


class SessionState(TypedDict):
    mode: Literal["learn", "interview"]
    messages: list[Turn]
    current_speaker: str           # "questioner" | "answerer"
    hitl_flags: list[str]          # 可包含 "questioner"、"answerer"
    turn_count: int
    max_turns: int
    context_strategy: str          # "auto" | "direct" | "rag"
    source_path: str
    outline: list[str]             # planner 生成的提问大纲，空列表表示尚未生成
    llm_reference_answer: str      # LLM 生成的参考答案，HITL 模式下供参考，非 HITL 时为空字符串
    review_history: list[list[KeywordScore]]  # review_history[i] = turn i keyword scores
```

- [ ] **Step 4: Update make_state() in tests/test_graph_nodes.py**

Add `llm_reference_answer` and `review_history` to defaults (they may already be missing — add both):

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_keyword_score_typeddict tests/test_graph_nodes.py::test_session_state_has_review_history_field tests/test_graph_nodes.py::test_session_state_review_history_accumulates -v
```

Expected: 3 PASS

- [ ] **Step 6: Run full test suite to check no regressions**

```bash
conda run -n agent pytest tests/test_graph_nodes.py -v
```

Expected: all existing tests still PASS

- [ ] **Step 7: Commit**

```bash
git add biteme/graph/state.py tests/test_graph_nodes.py
git commit -m "feat: add KeywordScore TypedDict and review_history to SessionState"
```

---

## Task 2: Update prompts.py

**Files:**
- Modify: `biteme/graph/prompts.py`
- Test: `tests/test_graph_nodes.py`

- [ ] **Step 1: Write the failing test**

```python
def test_get_prompts_learn_has_reviewer():
    prompts = get_prompts("learn")
    assert "reviewer" in prompts
    assert isinstance(prompts["reviewer"], str)
    assert len(prompts["reviewer"]) > 0

def test_get_prompts_interview_has_reviewer():
    prompts = get_prompts("interview")
    assert "reviewer" in prompts
    assert isinstance(prompts["reviewer"], str)
    assert len(prompts["reviewer"]) > 0

def test_reviewer_prompt_requests_json_output():
    prompts = get_prompts("learn")
    assert "JSON" in prompts["reviewer"]
    assert "keywords" in prompts["reviewer"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_get_prompts_learn_has_reviewer -v
```

Expected: FAIL with `AssertionError: 'reviewer' not in ...`

- [ ] **Step 3: Add REVIEWER constant and update get_prompts() in prompts.py**

Add after the existing constants (before `get_prompts`):

```python
REVIEWER = """\
你是一位严格的评审员。根据以下三段内容：
- 问题
- 用户回答
- LLM 参考答案

从问题中提炼 1–3 个核心知识点关键词，
对比用户回答与 LLM 参考答案，对每个关键词给出 0–10 的整数分。
评分综合考量：准确性（与参考答案是否一致）、完整性（要点是否遗漏）、深度（是否触及核心原理）。
10 分 = 与参考答案高度一致且有深度，0 分 = 与参考答案严重偏差或完全未提及。

只输出合法 JSON，格式如下，不得有任何其他文字：
{"keywords": [{"keyword": "...", "score": 0}]}
"""
```

Update `get_prompts()` to include `"reviewer"` in both branches:

```python
def get_prompts(mode: str) -> dict[str, str]:
    if mode == "learn":
        return {
            "questioner": LEARN_QUESTIONER,
            "answerer": LEARN_ANSWERER,
            "planner": LEARN_PLANNER,
            "reviewer": REVIEWER,
        }
    return {
        "questioner": INTERVIEW_QUESTIONER,
        "answerer": INTERVIEW_ANSWERER,
        "planner": INTERVIEW_PLANNER,
        "reviewer": REVIEWER,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_get_prompts_learn_has_reviewer tests/test_graph_nodes.py::test_get_prompts_interview_has_reviewer tests/test_graph_nodes.py::test_reviewer_prompt_requests_json_output -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add biteme/graph/prompts.py tests/test_graph_nodes.py
git commit -m "feat: add REVIEWER prompt to prompts.py"
```

---

## Task 3: Add reviewer_node to nodes.py

**Files:**
- Modify: `biteme/graph/nodes.py`
- Test: `tests/test_graph_nodes.py`

- [ ] **Step 1: Write the failing tests**

```python
from biteme.graph.nodes import reviewer_node

def test_reviewer_node_appends_to_review_history():
    """reviewer_node must append one list[KeywordScore] per call."""
    state = make_state(
        hitl_flags=["answerer"],
        messages=[
            {"speaker": "questioner", "content": "什么是 RAG？", "retrieved_chunks": []},
            {"speaker": "human", "content": "RAG 是检索增强生成。", "retrieved_chunks": []},
        ],
        llm_reference_answer="RAG（Retrieval-Augmented Generation）结合检索与生成，先从知识库检索相关片段，再输入 LLM 生成答案。核心组件：向量检索、embedding 模型、LLM。",
        review_history=[],
    )
    llm_response = '{"keywords": [{"keyword": "向量检索", "score": 7}, {"keyword": "embedding", "score": 5}]}'

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = llm_response

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.Console"):
        result = reviewer_node(state)

    assert "review_history" in result
    assert len(result["review_history"]) == 1
    turn_keywords = result["review_history"][0]
    assert len(turn_keywords) == 2
    assert turn_keywords[0]["keyword"] == "向量检索"
    assert turn_keywords[0]["score"] == 7


def test_reviewer_node_handles_malformed_json():
    """If LLM returns invalid JSON, reviewer_node appends an empty list (no crash)."""
    state = make_state(
        hitl_flags=["answerer"],
        messages=[
            {"speaker": "questioner", "content": "问题", "retrieved_chunks": []},
            {"speaker": "human", "content": "回答", "retrieved_chunks": []},
        ],
        llm_reference_answer="参考答案",
        review_history=[],
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "这不是JSON"

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.Console"):
        result = reviewer_node(state)

    assert result["review_history"] == [[]]


def test_reviewer_node_accumulates_across_turns():
    """review_history grows by one entry each time reviewer_node runs."""
    existing = [[{"keyword": "旧关键词", "score": 6}]]
    state = make_state(
        hitl_flags=["answerer"],
        messages=[
            {"speaker": "questioner", "content": "新问题", "retrieved_chunks": []},
            {"speaker": "human", "content": "新回答", "retrieved_chunks": []},
        ],
        llm_reference_answer="新参考",
        review_history=existing,
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"keywords": [{"keyword": "新关键词", "score": 9}]}'

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.Console"):
        result = reviewer_node(state)

    assert len(result["review_history"]) == 2
    assert result["review_history"][0][0]["keyword"] == "旧关键词"
    assert result["review_history"][1][0]["keyword"] == "新关键词"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_reviewer_node_appends_to_review_history -v
```

Expected: `ImportError: cannot import name 'reviewer_node'`

- [ ] **Step 3: Add reviewer_node to nodes.py**

Add `import json` at the top of `biteme/graph/nodes.py` (after existing imports).

Add `SystemMessage` to the existing `langchain_core.messages` import line:

```python
from langchain_core.messages import HumanMessage, SystemMessage
```

Add the following function at the end of `biteme/graph/nodes.py`:

```python
def reviewer_node(state: SessionState) -> dict:
    question = state["messages"][-2]["content"] if len(state["messages"]) >= 2 else ""
    user_answer = state["messages"][-1]["content"] if state["messages"] else ""
    llm_answer = state.get("llm_reference_answer", "")

    prompts = get_prompts(state["mode"])
    llm = ChatOpenAI(model=settings.openai_model, temperature=0.1)

    response = llm.invoke([
        SystemMessage(content=prompts["reviewer"]),
        HumanMessage(content=(
            f"问题：{question}\n\n"
            f"用户回答：{user_answer}\n\n"
            f"LLM 参考答案：{llm_answer}"
        )),
    ])

    try:
        data = json.loads(response.content)
        keywords: list = data.get("keywords", [])
    except (json.JSONDecodeError, AttributeError):
        keywords = []

    console = Console()

    def _color(score: int) -> str:
        if score >= 7:
            return "green"
        if score >= 4:
            return "yellow"
        return "red"

    lines = "\n".join(
        f"[{_color(kw['score'])}]{kw['keyword']}[/{_color(kw['score'])}]  {kw['score']}"
        for kw in keywords
    )
    console.print(Panel(lines or "[dim]（无关键词）[/dim]", title="[bold]本轮评审[/bold]"))

    return {
        "review_history": state["review_history"] + [keywords],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_reviewer_node_appends_to_review_history tests/test_graph_nodes.py::test_reviewer_node_handles_malformed_json tests/test_graph_nodes.py::test_reviewer_node_accumulates_across_turns -v
```

Expected: 3 PASS

- [ ] **Step 5: Run full test suite**

```bash
conda run -n agent pytest tests/test_graph_nodes.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add biteme/graph/nodes.py tests/test_graph_nodes.py
git commit -m "feat: add reviewer_node to nodes.py"
```

---

## Task 4: Update graph.py — routing and wiring

**Files:**
- Modify: `biteme/graph/graph.py`
- Test: `tests/test_graph_nodes.py`

- [ ] **Step 1: Write the failing tests**

```python
from biteme.graph.graph import _after_answerer

def test_after_answerer_routes_to_reviewer_when_hitl():
    state = make_state(hitl_flags=["answerer"], turn_count=2, max_turns=5)
    assert _after_answerer(state) == "reviewer"

def test_after_answerer_routes_to_end_when_max_turns_non_hitl():
    state = make_state(hitl_flags=[], turn_count=5, max_turns=5)
    from langgraph.graph import END
    assert _after_answerer(state) == END

def test_after_answerer_routes_to_questioner_otherwise():
    state = make_state(hitl_flags=[], turn_count=2, max_turns=5)
    assert _after_answerer(state) == "questioner"

def test_graph_has_reviewer_node():
    from biteme.graph.graph import build_graph
    graph = build_graph(checkpointer=None)
    assert "reviewer" in set(graph.nodes.keys())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_after_answerer_routes_to_reviewer_when_hitl -v
```

Expected: `ImportError: cannot import name '_after_answerer'`

- [ ] **Step 3: Rewrite graph.py**

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import SessionState
from .nodes import planner_node, questioner_node, answerer_node, reviewer_node


def _after_answerer(state: SessionState) -> str:
    if "answerer" in state["hitl_flags"]:
        return "reviewer"
    if state["turn_count"] >= state["max_turns"]:
        return END
    return "questioner"


def _should_continue(state: SessionState) -> str:
    if state["turn_count"] >= state["max_turns"]:
        return END
    return "questioner"


def build_graph(checkpointer: SqliteSaver) -> StateGraph:
    builder = StateGraph(SessionState)
    builder.add_node("planner", planner_node)
    builder.add_node("questioner", questioner_node)
    builder.add_node("answerer", answerer_node)
    builder.add_node("reviewer", reviewer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "questioner")
    builder.add_edge("questioner", "answerer")
    builder.add_conditional_edges(
        "answerer",
        _after_answerer,
        {"reviewer": "reviewer", "questioner": "questioner", END: END},
    )
    builder.add_conditional_edges(
        "reviewer",
        _should_continue,
        {"questioner": "questioner", END: END},
    )
    return builder.compile(checkpointer=checkpointer, interrupt_before=[])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_after_answerer_routes_to_reviewer_when_hitl tests/test_graph_nodes.py::test_after_answerer_routes_to_end_when_max_turns_non_hitl tests/test_graph_nodes.py::test_after_answerer_routes_to_questioner_otherwise tests/test_graph_nodes.py::test_graph_has_reviewer_node -v
```

Expected: 4 PASS

- [ ] **Step 5: Run full test suite**

```bash
conda run -n agent pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add biteme/graph/graph.py tests/test_graph_nodes.py
git commit -m "feat: wire reviewer_node into graph with _after_answerer routing"
```
