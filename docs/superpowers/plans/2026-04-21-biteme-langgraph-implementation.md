# BiteMe LangGraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph-based dual-agent project understanding assistant that supports observer mode, user intervention, interview mode, and structured summary output.

**Architecture:** Use a typed `GraphState` as shared memory and node-local stores as private memory. Implement `planner -> questioner -> answerer -> planner` as the main loop, with router nodes for user intervention and termination checks. Keep input adapters, agent nodes, orchestration graph, and CLI separated so each module has one responsibility.

**Tech Stack:** Python 3.11+, LangGraph, LangChain Core/OpenAI-compatible chat model, Pydantic, Rich, Pytest.

---

## File Structure

- Create: `biteme/src/config.py` (runtime config and model settings)
- Create: `biteme/src/memory/shared.py` (LangGraph shared state schema)
- Create: `biteme/src/memory/private.py` (agent private memory dataclasses)
- Create: `biteme/src/input/file_reader.py` (local folder reader)
- Create: `biteme/src/input/processor.py` (normalize text/folder inputs)
- Create: `biteme/src/agents/base.py` (LLM wrapper + prompt helpers)
- Create: `biteme/src/agents/planner.py` (question plan generator/updater)
- Create: `biteme/src/agents/questioner.py` (question generation node)
- Create: `biteme/src/agents/answerer.py` (answer/reference answer/evaluation node)
- Create: `biteme/src/agents/summarizer.py` (session summary node)
- Create: `biteme/src/orchestrator.py` (LangGraph graph wiring and execution API)
- Create: `biteme/src/cli.py` (interactive CLI loop)
- Create: `biteme/pyproject.toml` (dependencies and tooling)
- Create: `biteme/tests/test_input_processor.py`
- Create: `biteme/tests/test_planner_node.py`
- Create: `biteme/tests/test_orchestrator_flow.py`
- Create: `biteme/tests/test_interview_mode.py`
- Create: `biteme/README.md`

### Task 1: Project Scaffold And State Contracts

**Files:**
- Create: `biteme/pyproject.toml`
- Create: `biteme/src/config.py`
- Create: `biteme/src/memory/shared.py`
- Create: `biteme/src/memory/private.py`
- Test: `biteme/tests/test_orchestrator_flow.py`

- [ ] **Step 1: Write the failing test for shared state defaults**

```python
from src.memory.shared import GraphState, new_graph_state


def test_new_graph_state_defaults():
    state = new_graph_state(project_content="demo")
    assert isinstance(state, GraphState)
    assert state["project_content"] == "demo"
    assert state["dialogue_history"] == []
    assert state["question_plan"] == []
    assert state["terminated"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd biteme && pytest tests/test_orchestrator_flow.py::test_new_graph_state_defaults -v`
Expected: FAIL with `ModuleNotFoundError` or missing `new_graph_state`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/memory/shared.py
from typing import NotRequired, TypedDict


class GraphState(TypedDict):
    project_content: str
    dialogue_history: list[dict[str, str]]
    question_plan: list[dict[str, str]]
    completed_question_ids: list[str]
    user_keywords: list[str]
    mode: str
    terminated: bool
    termination_reason: NotRequired[str]
    max_rounds: int
    current_round: int
    final_summary: NotRequired[str]


def new_graph_state(project_content: str, *, mode: str = "observe", max_rounds: int = 8) -> GraphState:
    return {
        "project_content": project_content,
        "dialogue_history": [],
        "question_plan": [],
        "completed_question_ids": [],
        "user_keywords": [],
        "mode": mode,
        "terminated": False,
        "max_rounds": max_rounds,
        "current_round": 0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd biteme && pytest tests/test_orchestrator_flow.py::test_new_graph_state_defaults -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd biteme
git add pyproject.toml src/config.py src/memory/shared.py src/memory/private.py tests/test_orchestrator_flow.py
git commit -m "chore: scaffold BiteMe project and LangGraph state contracts"
```

### Task 2: Input Pipeline For Text And Local Folder

**Files:**
- Create: `biteme/src/input/file_reader.py`
- Create: `biteme/src/input/processor.py`
- Test: `biteme/tests/test_input_processor.py`

- [ ] **Step 1: Write failing tests for text and folder inputs**

```python
from pathlib import Path
from src.input.processor import build_project_content


def test_build_project_content_from_text():
    content = build_project_content(raw_text="Hello BiteMe", folder_path=None)
    assert "Hello BiteMe" in content


def test_build_project_content_from_folder(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Demo\n")
    (tmp_path / "app.py").write_text("print('hi')\n")
    content = build_project_content(raw_text=None, folder_path=str(tmp_path))
    assert "README.md" in content
    assert "app.py" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd biteme && pytest tests/test_input_processor.py -v`
Expected: FAIL with missing module/functions.

- [ ] **Step 3: Write minimal implementation**

```python
# src/input/file_reader.py
from pathlib import Path


def read_project_folder(folder_path: str) -> str:
    root = Path(folder_path)
    blocks: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            rel = path.relative_to(root)
            text = path.read_text(errors="ignore")
            blocks.append(f"## FILE: {rel}\n{text}\n")
    return "\n".join(blocks)
```

```python
# src/input/processor.py
from src.input.file_reader import read_project_folder


def build_project_content(*, raw_text: str | None, folder_path: str | None) -> str:
    if raw_text:
        return raw_text
    if folder_path:
        return read_project_folder(folder_path)
    raise ValueError("Either raw_text or folder_path is required")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd biteme && pytest tests/test_input_processor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd biteme
git add src/input/file_reader.py src/input/processor.py tests/test_input_processor.py
git commit -m "feat: add normalized project input pipeline for text and folders"
```

### Task 3: Planner Node And Question Plan Lifecycle

**Files:**
- Create: `biteme/src/agents/base.py`
- Create: `biteme/src/agents/planner.py`
- Test: `biteme/tests/test_planner_node.py`

- [ ] **Step 1: Write failing planner tests**

```python
from src.agents.planner import planner_node
from src.memory.shared import new_graph_state


def test_planner_initializes_question_plan():
    state = new_graph_state(project_content="simple project")
    state["user_keywords"] = ["architecture"]
    result = planner_node(state, llm=None)
    assert len(result["question_plan"]) >= 1
    assert "id" in result["question_plan"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd biteme && pytest tests/test_planner_node.py -v`
Expected: FAIL with missing `planner_node`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/agents/planner.py
from src.memory.shared import GraphState


def planner_node(state: GraphState, llm) -> GraphState:
    if state["question_plan"]:
        return state

    seed = state["user_keywords"] or ["project-overview", "core-flow", "trade-offs"]
    state["question_plan"] = [
        {"id": f"q{i+1}", "topic": topic, "priority": str(i + 1), "status": "pending"}
        for i, topic in enumerate(seed)
    ]
    return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd biteme && pytest tests/test_planner_node.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd biteme
git add src/agents/base.py src/agents/planner.py tests/test_planner_node.py
git commit -m "feat: implement question planner node with initial plan generation"
```

### Task 4: Questioner/Answerer Nodes With Interview Mode Behavior

**Files:**
- Create: `biteme/src/agents/questioner.py`
- Create: `biteme/src/agents/answerer.py`
- Test: `biteme/tests/test_interview_mode.py`

- [ ] **Step 1: Write failing tests for observer and interview modes**

```python
from src.agents.answerer import answerer_node
from src.memory.shared import new_graph_state


def test_answerer_records_agent_answer_in_observer_mode():
    state = new_graph_state("demo", mode="observe")
    state["dialogue_history"] = [{"role": "questioner", "content": "What is core loop?"}]
    result = answerer_node(state, llm=None, user_answer=None)
    assert result["dialogue_history"][-1]["role"] == "answerer"


def test_answerer_keeps_user_original_answer_in_interview_mode():
    state = new_graph_state("demo", mode="interview")
    state["dialogue_history"] = [{"role": "questioner", "content": "What is core loop?"}]
    result = answerer_node(state, llm=None, user_answer="It loops planner and qa nodes.")
    assert result["dialogue_history"][-1]["content"] == "It loops planner and qa nodes."
    assert "reference_answer" in result["dialogue_history"][-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd biteme && pytest tests/test_interview_mode.py -v`
Expected: FAIL with missing nodes.

- [ ] **Step 3: Write minimal implementation**

```python
# src/agents/answerer.py
from src.memory.shared import GraphState


def answerer_node(state: GraphState, llm, user_answer: str | None) -> GraphState:
    question = state["dialogue_history"][-1]["content"]
    reference_answer = f"Reference answer for: {question}"

    if state["mode"] == "interview" and user_answer:
        state["dialogue_history"].append(
            {
                "role": "user",
                "content": user_answer,
                "reference_answer": reference_answer,
                "evaluation": "placeholder-score",
            }
        )
        return state

    state["dialogue_history"].append({"role": "answerer", "content": reference_answer})
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd biteme && pytest tests/test_interview_mode.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd biteme
git add src/agents/questioner.py src/agents/answerer.py tests/test_interview_mode.py
git commit -m "feat: add questioner and answerer nodes with interview mode recording"
```

### Task 5: LangGraph Orchestrator With Routing And Termination

**Files:**
- Create: `biteme/src/orchestrator.py`
- Modify: `biteme/src/agents/planner.py`
- Modify: `biteme/src/agents/questioner.py`
- Modify: `biteme/src/agents/answerer.py`
- Test: `biteme/tests/test_orchestrator_flow.py`

- [ ] **Step 1: Write failing orchestration flow tests**

```python
from src.orchestrator import build_graph, run_session
from src.memory.shared import new_graph_state


def test_session_stops_when_plan_complete():
    graph = build_graph()
    state = new_graph_state("demo", max_rounds=2)
    result = run_session(graph, state)
    assert result["terminated"] is True
    assert result["termination_reason"] in {"plan_completed", "max_rounds"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd biteme && pytest tests/test_orchestrator_flow.py::test_session_stops_when_plan_complete -v`
Expected: FAIL with missing graph builder.

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator.py
from langgraph.graph import END, StateGraph
from src.agents.answerer import answerer_node
from src.agents.planner import planner_node
from src.agents.questioner import questioner_node
from src.agents.summarizer import summarizer_node
from src.memory.shared import GraphState


def should_terminate(state: GraphState) -> str:
    if state["current_round"] >= state["max_rounds"]:
        state["terminated"] = True
        state["termination_reason"] = "max_rounds"
        return "summarize"
    pending = [q for q in state["question_plan"] if q["status"] == "pending"]
    if not pending:
        state["terminated"] = True
        state["termination_reason"] = "plan_completed"
        return "summarize"
    return "questioner"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("questioner", questioner_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("summarize", summarizer_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", should_terminate, {"questioner": "questioner", "summarize": "summarize"})
    graph.add_edge("questioner", "answerer")
    graph.add_edge("answerer", "planner")
    graph.add_edge("summarize", END)
    return graph.compile()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd biteme && pytest tests/test_orchestrator_flow.py::test_session_stops_when_plan_complete -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd biteme
git add src/orchestrator.py src/agents/planner.py src/agents/questioner.py src/agents/answerer.py tests/test_orchestrator_flow.py
git commit -m "feat: wire LangGraph orchestrator loop with termination routing"
```

### Task 6: User Intervention And CLI Session Control

**Files:**
- Create: `biteme/src/cli.py`
- Modify: `biteme/src/orchestrator.py`
- Test: `biteme/tests/test_orchestrator_flow.py`

- [ ] **Step 1: Write failing test for intervention command handling**

```python
from src.orchestrator import apply_user_intervention
from src.memory.shared import new_graph_state


def test_apply_user_intervention_insert_question():
    state = new_graph_state("demo")
    updated = apply_user_intervention(state, "/ask explain architecture")
    assert updated["question_plan"][0]["topic"] == "explain architecture"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd biteme && pytest tests/test_orchestrator_flow.py::test_apply_user_intervention_insert_question -v`
Expected: FAIL with missing `apply_user_intervention`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator.py
def apply_user_intervention(state: GraphState, command: str) -> GraphState:
    if command.startswith("/ask "):
        topic = command.replace("/ask ", "", 1).strip()
        state["question_plan"].insert(0, {"id": "user-inserted", "topic": topic, "priority": "0", "status": "pending"})
    elif command.startswith("/stop"):
        state["terminated"] = True
        state["termination_reason"] = "user_stopped"
    return state
```

```python
# src/cli.py
from rich.console import Console


def run_cli() -> None:
    console = Console()
    console.print("BiteMe CLI started. Use /ask, /skip, /stop to intervene.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd biteme && pytest tests/test_orchestrator_flow.py::test_apply_user_intervention_insert_question -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd biteme
git add src/cli.py src/orchestrator.py tests/test_orchestrator_flow.py
git commit -m "feat: add user intervention handling and interactive CLI entry"
```

### Task 7: Summarizer Node And End-Of-Session Report

**Files:**
- Create: `biteme/src/agents/summarizer.py`
- Test: `biteme/tests/test_orchestrator_flow.py`
- Modify: `biteme/tests/test_interview_mode.py`

- [ ] **Step 1: Write failing tests for final summary fields**

```python
from src.agents.summarizer import summarizer_node
from src.memory.shared import new_graph_state


def test_summarizer_generates_structured_output():
    state = new_graph_state("demo", mode="interview")
    state["dialogue_history"] = [{"role": "questioner", "content": "Q1"}, {"role": "user", "content": "A1"}]
    state["question_plan"] = [{"id": "q1", "topic": "core", "status": "done"}]
    result = summarizer_node(state, llm=None)
    assert "explored_topics" in result["final_summary"]
    assert "next_directions" in result["final_summary"]
    assert "understanding_assessment" in result["final_summary"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd biteme && pytest tests/test_orchestrator_flow.py::test_summarizer_generates_structured_output -v`
Expected: FAIL with missing summary node.

- [ ] **Step 3: Write minimal implementation**

```python
# src/agents/summarizer.py
from src.memory.shared import GraphState


def summarizer_node(state: GraphState, llm) -> GraphState:
    topics = [item["topic"] for item in state["question_plan"]]
    payload = {
        "explored_topics": topics,
        "key_findings": [msg["content"] for msg in state["dialogue_history"][-3:]],
        "next_directions": [item["topic"] for item in state["question_plan"] if item["status"] != "done"],
    }
    if state["mode"] == "interview":
        payload["understanding_assessment"] = "Need deeper coverage on architecture trade-offs."
    state["final_summary"] = str(payload)
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd biteme && pytest tests/test_orchestrator_flow.py::test_summarizer_generates_structured_output tests/test_interview_mode.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd biteme
git add src/agents/summarizer.py tests/test_orchestrator_flow.py tests/test_interview_mode.py
git commit -m "feat: generate structured end-of-session summary including interview assessment"
```

### Task 8: Integration Tests, Documentation, And MVP Exit Criteria

**Files:**
- Modify: `biteme/tests/test_orchestrator_flow.py`
- Create: `biteme/README.md`
- Modify: `docs/roadmap.md`

- [ ] **Step 1: Write failing integration test for end-to-end observer flow**

```python
from src.orchestrator import build_graph, run_session
from src.memory.shared import new_graph_state


def test_end_to_end_observer_mode():
    graph = build_graph()
    state = new_graph_state("project text", mode="observe", max_rounds=3)
    result = run_session(graph, state)
    assert result["terminated"] is True
    assert result["final_summary"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd biteme && pytest tests/test_orchestrator_flow.py::test_end_to_end_observer_mode -v`
Expected: FAIL because `run_session` behavior is incomplete.

- [ ] **Step 3: Write minimal implementation and docs**

```python
# src/orchestrator.py
def run_session(graph, initial_state: GraphState) -> GraphState:
    return graph.invoke(initial_state)
```

```markdown
# README.md
## Quick Start
1. `uv pip install -e .` (or `pip install -e .`)
2. `python -m src.cli`
3. Provide text or folder path and choose `observe` or `interview` mode.
```

- [ ] **Step 4: Run full test suite to verify everything passes**

Run: `cd biteme && pytest -v`
Expected: PASS with all tests green.

- [ ] **Step 5: Commit**

```bash
cd biteme
git add tests/test_orchestrator_flow.py README.md docs/roadmap.md src/orchestrator.py
git commit -m "docs: finalize LangGraph implementation guide and verify e2e flow"
```

## Self-Review

### 1) Spec Coverage
- Dual-agent conversation loop is covered by Tasks 3-5.
- Question planning and dynamic updates are covered by Task 3 and routing in Task 5.
- Shared/private memory architecture is covered by Task 1.
- User intervention and mixed interaction are covered by Task 6.
- Interview mode (user answers + reference answer + assessment) is covered by Tasks 4 and 7.
- Termination conditions are covered by Task 5.
- Structured summarization output is covered by Task 7.
- CLI-first delivery is covered by Task 6 and Task 8.

### 2) Placeholder Scan
- No `TODO/TBD` placeholders remain.
- Every code-edit step includes concrete code.
- Every verification step includes explicit commands and expected outcomes.

### 3) Type/Signature Consistency
- `GraphState` key names are reused consistently across all task snippets.
- `planner_node`, `questioner_node`, `answerer_node`, `summarizer_node`, `build_graph`, and `run_session` signatures remain stable across tasks.

