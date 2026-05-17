# Planner Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 questioner/answerer 循环前插入一个固定执行的 planner 节点，根据文档内容和 max_turns 生成提问大纲，展示给用户，并供 questioner 参考。

**Architecture:** 新增 `planner_node` 作为 LangGraph 图的入口节点，生成的大纲存入 `SessionState.outline`，随 checkpoint 持久化。questioner AI 模式下将大纲附加到 prompt；HITL 模式下按轮次预填建议问题。

**Tech Stack:** LangGraph, LangChain, OpenAI API, Rich, Typer, pytest + unittest.mock

---

## File Structure

| 文件 | 操作 | 说明 |
|------|------|------|
| `biteme/graph/state.py` | 修改 | 新增 `outline: list[str]` 字段 |
| `biteme/graph/prompts.py` | 修改 | 新增 `LEARN_PLANNER`、`INTERVIEW_PLANNER`；扩展 `get_prompts()` |
| `biteme/graph/nodes.py` | 修改 | 新增 `planner_node` 和 `_parse_outline`；更新 `questioner_node` |
| `biteme/graph/graph.py` | 修改 | 新增 planner 节点和边；更改入口点 |
| `biteme/cli.py` | 修改 | `initial_state` 增加 `outline: []`；新增 `_extract_suggestion`；处理空回车 |
| `tests/test_graph_nodes.py` | 修改 | 更新 `make_state` helper；新增 planner 和 questioner outline 测试 |

---

## Task 1: 扩展 SessionState 并更新测试 helper

**Files:**
- Modify: `biteme/graph/state.py`
- Modify: `tests/test_graph_nodes.py`

- [ ] **Step 1: 写一个验证新字段存在的测试**

```python
# 在 tests/test_graph_nodes.py 顶部 import 后添加
def test_session_state_has_outline_field():
    state = make_state(outline=["Q1", "Q2"])
    assert state["outline"] == ["Q1", "Q2"]

def test_session_state_outline_defaults_to_empty():
    state = make_state()
    assert state["outline"] == []
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_session_state_has_outline_field tests/test_graph_nodes.py::test_session_state_outline_defaults_to_empty -v
```

Expected: FAIL（`make_state` 不含 `outline`，`SessionState` 无该字段）

- [ ] **Step 3: 更新 `biteme/graph/state.py`**

将文件改为：

```python
from typing import Literal
from typing_extensions import TypedDict


class Turn(TypedDict):
    speaker: str            # "questioner" | "answerer" | "human"
    content: str
    retrieved_chunks: list[str]  # answerer 轮携带，其余为空列表


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
```

- [ ] **Step 4: 更新 `tests/test_graph_nodes.py` 的 `make_state` helper**

将 `make_state` 函数改为：

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
    )
    defaults.update(kwargs)
    return defaults  # type: ignore
```

- [ ] **Step 5: 运行所有节点测试，确认通过**

```bash
conda run -n agent pytest tests/test_graph_nodes.py -v
```

Expected: 所有测试 PASS

- [ ] **Step 6: Commit**

```bash
git add biteme/graph/state.py tests/test_graph_nodes.py
git commit -m "feat: add outline field to SessionState"
```

---

## Task 2: 新增 Planner Prompts

**Files:**
- Modify: `biteme/graph/prompts.py`
- Modify: `tests/test_graph_nodes.py`（新增 prompt 测试）

- [ ] **Step 1: 写测试确认 get_prompts 返回 planner key**

在 `tests/test_graph_nodes.py` 中添加：

```python
from biteme.graph.prompts import get_prompts

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
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_get_prompts_learn_has_planner tests/test_graph_nodes.py::test_get_prompts_interview_has_planner -v
```

Expected: FAIL（`get_prompts` 不含 `"planner"` key）

- [ ] **Step 3: 更新 `biteme/graph/prompts.py`**

将文件改为（完整内容）：

```python
LEARN_QUESTIONER = """\
你是一位好奇的学习者，正在探索【参考内容摘要】中提供的文档。
每次提出一个清晰、具体的问题，帮助自己逐步理解文档的结构、设计与实现。

严格规则：
- 问题必须直接来源于【参考内容摘要】中出现的内容，不得追问文档未涉及的概念。
- 若上一轮回答提到了文档之外的概念，忽略它，继续围绕文档本身提问。
- 不要重复已经问过的问题。
- 只输出问题本身，不要加任何前缀或解释。
"""

LEARN_ANSWERER = """\
你是内容专家。基于以下检索到的相关片段，详细回答问题。
如果片段中有文件路径，请在回答中引用它们。
如果检索片段不足以完整回答，请诚实说明。

相关内容：
{context}
"""

LEARN_PLANNER = """\
你是一位学习规划者。根据提供的文档摘要，为学习者规划由浅入深的学习问题。
要求：
- 覆盖文档的主要知识点
- 问题之间有逻辑递进关系
- 只输出问题本身，不要加任何解释或前缀
- 按编号列出，格式：1. xxx  2. xxx ...
"""

INTERVIEW_QUESTIONER = """\
你是一位经验丰富的技术面试官，面试内容严格限定在【参考内容摘要】所描述的技术范围内。
每轮提出一个有深度的技术问题，并在问题前简短评价上一轮的回答（第一轮跳过评价）。

严格规则：
- 问题必须有明确的文档依据，只能考察【参考内容摘要】中涉及的技术点。
- 若候选人的回答引申出文档未涉及的内容，忽略该引申，继续围绕文档本身出题。
- 只输出评价（若有）+ 问题，不要加其他前缀。
"""

INTERVIEW_ANSWERER = """\
你是一位技术面试候选人。给出简洁、专业的技术回答。
如有必要，可参考以下检索到的相关背景信息，但不要直接照抄。

参考信息：
{context}
"""

INTERVIEW_PLANNER = """\
你是一位技术面试规划者。根据提供的文档摘要，为技术面试规划考察问题，从基础到深度递进。
要求：
- 严格基于文档内容出题
- 覆盖核心技术点，包括设计决策、实现细节、潜在问题
- 只输出问题本身，不要加任何解释或前缀
- 按编号列出，格式：1. xxx  2. xxx ...
"""


def get_prompts(mode: str) -> dict[str, str]:
    if mode == "learn":
        return {
            "questioner": LEARN_QUESTIONER,
            "answerer": LEARN_ANSWERER,
            "planner": LEARN_PLANNER,
        }
    return {
        "questioner": INTERVIEW_QUESTIONER,
        "answerer": INTERVIEW_ANSWERER,
        "planner": INTERVIEW_PLANNER,
    }
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_get_prompts_learn_has_planner tests/test_graph_nodes.py::test_get_prompts_interview_has_planner -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add biteme/graph/prompts.py tests/test_graph_nodes.py
git commit -m "feat: add planner prompts for learn and interview modes"
```

---

## Task 3: 实现 `_parse_outline` 和 `planner_node`

**Files:**
- Modify: `biteme/graph/nodes.py`
- Modify: `tests/test_graph_nodes.py`

- [ ] **Step 1: 写 `_parse_outline` 的单元测试**

在 `tests/test_graph_nodes.py` 中添加：

```python
from biteme.graph.nodes import _parse_outline

def test_parse_outline_strips_numbering():
    text = "1. 什么是架构设计？\n2. 如何处理异常？\n3. 性能优化有哪些方法？"
    result = _parse_outline(text)
    assert result == ["什么是架构设计？", "如何处理异常？", "性能优化有哪些方法？"]

def test_parse_outline_ignores_empty_lines():
    text = "1. 问题一\n\n2. 问题二\n\n"
    result = _parse_outline(text)
    assert result == ["问题一", "问题二"]

def test_parse_outline_handles_parenthesis_numbering():
    text = "1) 问题一\n2) 问题二"
    result = _parse_outline(text)
    assert result == ["问题一", "问题二"]
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_parse_outline_strips_numbering tests/test_graph_nodes.py::test_parse_outline_ignores_empty_lines tests/test_graph_nodes.py::test_parse_outline_handles_parenthesis_numbering -v
```

Expected: FAIL（`_parse_outline` 未定义）

- [ ] **Step 3: 写 `planner_node` 的测试**

在 `tests/test_graph_nodes.py` 中添加：

```python
from biteme.graph.nodes import planner_node

def test_planner_node_populates_outline(tmp_path):
    state = make_state(source_path=str(tmp_path), max_turns=3)
    (tmp_path / "doc.md").write_text("# BiteMe\n双Agent问答系统。")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        "1. 什么是BiteMe？\n2. 它有哪些核心功能？\n"
        "3. 如何启动会话？\n4. HITL如何工作？\n5. 如何恢复会话？"
    )

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory, \
         patch("biteme.graph.nodes.Console"):
        mock_provider = MagicMock()
        mock_provider.get_overview.return_value = ["# BiteMe\n双Agent问答系统。"]
        mock_factory.return_value = mock_provider
        result = planner_node(state)

    assert len(result["outline"]) == 5  # max_turns(3) + 2
    assert result["outline"][0] == "什么是BiteMe？"
    mock_provider.get_overview.assert_called_once()

def test_planner_node_interview_mode(tmp_path):
    state = make_state(source_path=str(tmp_path), max_turns=2, mode="interview")
    (tmp_path / "doc.md").write_text("# API Design")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "1. 接口设计原则是什么？\n2. 如何处理版本兼容？\n3. 错误码规范？\n4. 认证机制？"

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory, \
         patch("biteme.graph.nodes.Console"):
        mock_provider = MagicMock()
        mock_provider.get_overview.return_value = ["# API Design"]
        mock_factory.return_value = mock_provider
        result = planner_node(state)

    assert len(result["outline"]) == 4
    assert "outline" in result
```

- [ ] **Step 4: 运行测试，确认失败**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_planner_node_populates_outline tests/test_graph_nodes.py::test_planner_node_interview_mode -v
```

Expected: FAIL（`planner_node` 未定义）

- [ ] **Step 5: 在 `biteme/graph/nodes.py` 中实现 `_parse_outline` 和 `planner_node`**

在文件顶部 import 区域添加 `import re`。在现有 `_get_db_path` 函数之后、`questioner_node` 之前插入：

```python
import re


def _parse_outline(text: str) -> list[str]:
    """Parse numbered list from LLM output into plain question strings."""
    questions = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
        if cleaned:
            questions.append(cleaned)
    return questions


def planner_node(state: SessionState) -> dict:
    from rich.console import Console
    from rich.panel import Panel

    prompts = get_prompts(state["mode"])
    n = state["max_turns"] + 2
    provider = create_provider(
        source_path=state["source_path"],
        strategy=state["context_strategy"],
        db_path=_get_db_path(state["source_path"]),
    )
    overview_chunks = provider.get_overview()
    context_text = "\n\n---\n\n".join(overview_chunks[:3])

    llm = ChatOpenAI(model=settings.openai_model, temperature=0.7)
    response = llm.invoke([
        SystemMessage(content=prompts["planner"]),
        HumanMessage(content=f"文档摘要：\n{context_text[:3000]}\n\n请生成 {n} 个问题。"),
    ])

    outline = _parse_outline(response.content)

    title = "提问大纲" if state["mode"] == "learn" else "面试大纲"
    outline_display = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(outline))
    console = Console()
    console.print(Panel(outline_display, title=f"[cyan]{title}[/cyan]"))

    return {"outline": outline}
```

- [ ] **Step 6: 运行所有 planner 相关测试，确认通过**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_parse_outline_strips_numbering tests/test_graph_nodes.py::test_parse_outline_ignores_empty_lines tests/test_graph_nodes.py::test_parse_outline_handles_parenthesis_numbering tests/test_graph_nodes.py::test_planner_node_populates_outline tests/test_graph_nodes.py::test_planner_node_interview_mode -v
```

Expected: 全部 PASS

- [ ] **Step 7: 运行全部节点测试，确认无回归**

```bash
conda run -n agent pytest tests/test_graph_nodes.py -v
```

Expected: 全部 PASS

- [ ] **Step 8: Commit**

```bash
git add biteme/graph/nodes.py tests/test_graph_nodes.py
git commit -m "feat: implement planner_node and _parse_outline"
```

---

## Task 4: 更新图拓扑

**Files:**
- Modify: `biteme/graph/graph.py`
- Modify: `tests/test_graph_nodes.py`（新增图结构测试）

- [ ] **Step 1: 写图结构测试**

在 `tests/test_graph_nodes.py` 中添加：

```python
from unittest.mock import MagicMock

def test_graph_entry_is_planner():
    from biteme.graph.graph import build_graph
    mock_checkpointer = MagicMock()
    graph = build_graph(checkpointer=mock_checkpointer)
    # LangGraph compiled graph 的节点集合
    node_names = set(graph.nodes.keys())
    assert "planner" in node_names
    assert "questioner" in node_names
    assert "answerer" in node_names
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_graph_entry_is_planner -v
```

Expected: FAIL（`graph.nodes` 中无 `"planner"`）

- [ ] **Step 3: 更新 `biteme/graph/graph.py`**

将文件改为（完整内容）：

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import SessionState
from .nodes import planner_node, questioner_node, answerer_node


def _should_continue(state: SessionState) -> str:
    if state["turn_count"] >= state["max_turns"]:
        return END
    return "continue"


def build_graph(checkpointer: SqliteSaver) -> StateGraph:
    builder = StateGraph(SessionState)
    builder.add_node("planner", planner_node)
    builder.add_node("questioner", questioner_node)
    builder.add_node("answerer", answerer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "questioner")
    builder.add_edge("questioner", "answerer")
    builder.add_conditional_edges(
        "answerer",
        _should_continue,
        {"continue": "questioner", END: END},
    )
    return builder.compile(checkpointer=checkpointer, interrupt_before=[])
```

- [ ] **Step 4: 运行图结构测试，确认通过**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_graph_entry_is_planner -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add biteme/graph/graph.py tests/test_graph_nodes.py
git commit -m "feat: set planner as graph entry point"
```

---

## Task 5: Questioner 节点使用大纲

**Files:**
- Modify: `biteme/graph/nodes.py`（更新 `questioner_node`）
- Modify: `tests/test_graph_nodes.py`

- [ ] **Step 1: 写测试验证 AI questioner 将大纲附加到 prompt**

在 `tests/test_graph_nodes.py` 中添加：

```python
def test_questioner_includes_outline_in_prompt(tmp_path):
    outline = ["问题一：架构设计", "问题二：数据流"]
    state = make_state(source_path=str(tmp_path), outline=outline)
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "What does foo do?"

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.get_overview.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        questioner_node(state)

    call_args = mock_llm.invoke.call_args[0][0]
    human_message_content = call_args[1].content
    assert "提问大纲" in human_message_content
    assert "问题一：架构设计" in human_message_content

def test_questioner_no_outline_section_when_empty(tmp_path):
    state = make_state(source_path=str(tmp_path), outline=[])
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "What does foo do?"

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.get_overview.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        questioner_node(state)

    call_args = mock_llm.invoke.call_args[0][0]
    human_message_content = call_args[1].content
    assert "提问大纲" not in human_message_content
```

- [ ] **Step 2: 写 HITL questioner 建议问题测试**

```python
def test_questioner_hitl_uses_outline_as_suggestion(tmp_path):
    outline = ["第一个建议问题", "第二个建议问题"]
    state = make_state(
        source_path=str(tmp_path),
        hitl_flags=["questioner"],
        outline=outline,
        turn_count=0,
    )

    interrupted_value = None

    def fake_interrupt(msg):
        nonlocal interrupted_value
        interrupted_value = msg
        raise Exception("interrupt_called")  # 模拟 interrupt 中断执行

    with patch("biteme.graph.nodes.interrupt", side_effect=fake_interrupt):
        try:
            questioner_node(state)
        except Exception:
            pass

    assert interrupted_value is not None
    assert "第一个建议问题" in interrupted_value

def test_questioner_hitl_fallback_when_outline_exhausted(tmp_path):
    state = make_state(
        source_path=str(tmp_path),
        hitl_flags=["questioner"],
        outline=["唯一问题"],
        turn_count=5,  # 超出 outline 范围
    )

    interrupted_value = None

    def fake_interrupt(msg):
        nonlocal interrupted_value
        interrupted_value = msg
        raise Exception("interrupt_called")

    with patch("biteme.graph.nodes.interrupt", side_effect=fake_interrupt):
        try:
            questioner_node(state)
        except Exception:
            pass

    # 超出范围时使用默认提示，不报错
    assert interrupted_value is not None
    assert "提问者" in interrupted_value
```

- [ ] **Step 3: 运行新测试，确认失败**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_questioner_includes_outline_in_prompt tests/test_graph_nodes.py::test_questioner_no_outline_section_when_empty tests/test_graph_nodes.py::test_questioner_hitl_uses_outline_as_suggestion tests/test_graph_nodes.py::test_questioner_hitl_fallback_when_outline_exhausted -v
```

Expected: FAIL

- [ ] **Step 4: 更新 `biteme/graph/nodes.py` 中的 `questioner_node`**

将 `questioner_node` 函数替换为：

```python
def questioner_node(state: SessionState) -> dict:
    if "questioner" in state["hitl_flags"]:
        outline = state.get("outline", [])
        turn_idx = state["turn_count"]
        if outline and turn_idx < len(outline):
            suggested = outline[turn_idx]
            prompt_msg = (
                f">>> [提问者] 建议问题（第 {turn_idx + 1} 轮）：{suggested}\n"
                f"（直接回车使用建议问题，或输入新问题）"
            )
        else:
            prompt_msg = ">>> [提问者] 请输入你的问题："
        human_text = interrupt(prompt_msg)
        turn: Turn = {"speaker": "human", "content": human_text, "retrieved_chunks": []}
    else:
        prompts = get_prompts(state["mode"])
        history = "\n".join(
            f"[{t['speaker']}]: {t['content']}" for t in state["messages"][-6:]
        )
        provider = create_provider(
            source_path=state["source_path"],
            strategy=state["context_strategy"],
            db_path=_get_db_path(state["source_path"]),
        )
        if not state["messages"]:
            context_chunks = provider.get_overview()
        else:
            retrieval_query = state["messages"][-1]["content"]
            context_chunks = provider.retrieve(retrieval_query)
        context_text = "\n\n---\n\n".join(context_chunks[:3])

        outline = state.get("outline", [])
        outline_section = ""
        if outline:
            outline_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(outline))
            outline_section = f"\n\n提问大纲（供参考，请结合对话历史灵活使用，不必按顺序）：\n{outline_text}"

        llm = ChatOpenAI(model=settings.openai_model, temperature=0.7)
        response = llm.invoke([
            SystemMessage(content=prompts["questioner"]),
            HumanMessage(
                content=(
                    f"对话历史：\n{history}"
                    f"{outline_section}"
                    f"\n\n参考内容摘要：\n{context_text[:2000]}"
                    f"\n\n请提出下一个问题。"
                )
            ),
        ])
        turn = {"speaker": "questioner", "content": response.content, "retrieved_chunks": []}

    return {
        "messages": state["messages"] + [turn],
        "current_speaker": "answerer",
        "turn_count": state["turn_count"] + 1,
    }
```

- [ ] **Step 5: 运行所有节点测试，确认通过**

```bash
conda run -n agent pytest tests/test_graph_nodes.py -v
```

Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add biteme/graph/nodes.py tests/test_graph_nodes.py
git commit -m "feat: questioner uses outline as reference in AI and HITL modes"
```

---

## Task 6: CLI 更新

**Files:**
- Modify: `biteme/cli.py`

- [ ] **Step 1: 在 `biteme/cli.py` 中做以下三处修改**

**修改 1**：在 `run` 命令的 `initial_state` 中新增 `outline` 字段。

找到：
```python
    initial_state: SessionState = {
        "mode": mode,  # type: ignore
        "messages": [],
        "current_speaker": "questioner",
        "hitl_flags": hitl_flags,
        "turn_count": 0,
        "max_turns": turns,
        "context_strategy": strategy,
        "source_path": str(path.resolve()),
    }
```

改为：
```python
    initial_state: SessionState = {
        "mode": mode,  # type: ignore
        "messages": [],
        "current_speaker": "questioner",
        "hitl_flags": hitl_flags,
        "turn_count": 0,
        "max_turns": turns,
        "context_strategy": strategy,
        "source_path": str(path.resolve()),
        "outline": [],
    }
```

**修改 2**：在 `_print_turn` 函数之前，新增 `_extract_suggestion` 辅助函数：

```python
def _extract_suggestion(prompt_text: str) -> str:
    """从 HITL 提示文字中提取建议问题。
    
    提示格式：'>>> [提问者] 建议问题（第 N 轮）：{question}\\n（直接回车使用建议问题，或输入新问题）'
    """
    if "建议问题" in prompt_text and "：" in prompt_text:
        parts = prompt_text.split("：", 1)
        if len(parts) > 1:
            suggestion = parts[1].split("\n")[0].strip()
            return suggestion
    return ""
```

**修改 3**：在 `_run_graph` 函数中，更新 interrupt 处理逻辑，支持空回车使用建议问题。

找到：
```python
                user_input = typer.prompt("", prompt_suffix="> ")
                pending = Command(resume=user_input)
```

改为：
```python
                user_input = typer.prompt("", prompt_suffix="> ")
                if not user_input.strip():
                    user_input = _extract_suggestion(prompt_text)
                pending = Command(resume=user_input)
```

- [ ] **Step 2: 运行现有 CLI 测试，确认无回归**

```bash
conda run -n agent pytest tests/test_cli.py -v
```

Expected: 全部 PASS

- [ ] **Step 3: 运行完整测试套件**

```bash
conda run -n agent pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 4: Commit**

```bash
git add biteme/cli.py
git commit -m "feat: update CLI for planner - add outline to initial state and empty-enter suggestion support"
```

---

## Task 7: 最终集成验证

- [ ] **Step 1: 运行完整测试套件，确认一切正常**

```bash
conda run -n agent pytest tests/ -v --tb=short
```

Expected: 全部 PASS，无任何 WARNING 或 ERROR

- [ ] **Step 2: 快速手动冒烟测试（可选，需要有效的 OPENAI_API_KEY）**

```bash
# 准备一个小测试文档
echo "# 测试文档\n这是一个简单的测试文档，包含基本概念。" > /tmp/test_doc.md

# 运行 learn 模式（纯 AI，turns=2）
conda run -n agent biteme run /tmp/test_doc.md --mode learn --turns 2

# 确认：
# 1. 程序先展示带编号的提问大纲 Panel
# 2. 然后进入正常的 questioner/answerer 循环
```

- [ ] **Step 3: 最终 commit（如有遗漏的改动）**

```bash
git add -A
git status  # 确认没有遗漏的文件
# 如有未提交的改动：
git commit -m "chore: finalize planner agent integration"
```
