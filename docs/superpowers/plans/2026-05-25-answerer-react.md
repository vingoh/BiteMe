# Answerer ReAct 改造实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `answerer_node` 的 AI 回答分支从单次 LLM 调用改为 ReAct agent，使其能在回答过程中自主调用只读工具补充信息。

**Architecture:** 使用 `langgraph.prebuilt.create_react_agent` 在 `answerer_node` 内部构建 ReAct 子图，替代原来的 `llm.invoke()` 单次调用。主图结构不变，HITL 分支不变。新增 Tavily 联网搜索工具和 `READONLY_TOOLS` 导出。

**Tech Stack:** LangGraph (`create_react_agent`)、`langchain-tavily` (TavilySearch)、既有的 `biteme.tools` 模块

---

## 文件结构

| 文件 | 操作 | 职责 |
|---|---|---|
| `biteme/tools/web.py` | 新建 | 封装 TavilySearch 工具实例 |
| `biteme/tools/__init__.py` | 修改 | 导入 web 工具，新增 `READONLY_TOOLS` 导出 |
| `biteme/config.py` | 修改 | 新增 `tavily_api_key` 配置项 |
| `biteme/graph/prompts.py` | 修改 | 重写 answerer prompt，去掉 `{context}` 占位符，加工具使用指引 |
| `biteme/graph/nodes.py` | 修改 | `answerer_node` AI 分支改用 `create_react_agent` |
| `tests/test_tools_web.py` | 新建 | web 工具的单元测试 |
| `tests/test_graph_nodes.py` | 修改 | 更新 answerer 测试以适配 ReAct 调用方式 |

---

### Task 1: 安装依赖

**Files:**
- Modify: `pyproject.toml` 或项目依赖配置

- [ ] **Step 1: 安装 langchain-tavily**

```bash
conda run -n agent pip install langchain-tavily
```

- [ ] **Step 2: 验证安装**

```bash
conda run -n agent python -c "from langchain_tavily import TavilySearch; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "chore: add langchain-tavily dependency"
```

---

### Task 2: 新增 Tavily 搜索工具

**Files:**
- Create: `biteme/tools/web.py`
- Test: `tests/test_tools_web.py`

- [ ] **Step 1: 写测试**

在 `tests/test_tools_web.py` 中：

```python
from unittest.mock import patch, MagicMock


def test_tavily_search_is_importable():
    from biteme.tools.web import tavily_search
    assert tavily_search is not None
    assert tavily_search.name == "tavily_search"


def test_tavily_search_in_readonly_tools():
    from biteme.tools import READONLY_TOOLS
    names = [t.name for t in READONLY_TOOLS]
    assert "tavily_search" in names


def test_write_file_not_in_readonly_tools():
    from biteme.tools import READONLY_TOOLS
    names = [t.name for t in READONLY_TOOLS]
    assert "write_file" not in names
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n agent pytest tests/test_tools_web.py -v
```

Expected: FAIL（`biteme.tools.web` 不存在）

- [ ] **Step 3: 创建 `biteme/tools/web.py`**

```python
from langchain_tavily import TavilySearch

tavily_search = TavilySearch(max_results=3)
```

- [ ] **Step 4: 修改 `biteme/tools/__init__.py`**

替换整个文件内容为：

```python
from .filesystem import read_file, write_file, search_files_by_name, search_files_by_content
from .github import github_list_tree, github_read_file, github_search_code
from .web import tavily_search

ALL_TOOLS = [
    github_list_tree,
    github_read_file,
    github_search_code,
    read_file,
    write_file,
    search_files_by_name,
    search_files_by_content,
    tavily_search,
]

READONLY_TOOLS = [t for t in ALL_TOOLS if t.name != "write_file"]
```

- [ ] **Step 5: 运行测试确认通过**

```bash
conda run -n agent pytest tests/test_tools_web.py -v
```

Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add biteme/tools/web.py biteme/tools/__init__.py tests/test_tools_web.py
git commit -m "feat: add Tavily search tool and READONLY_TOOLS export"
```

---

### Task 3: 新增 `tavily_api_key` 配置

**Files:**
- Modify: `biteme/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写测试**

在 `tests/test_config.py` 末尾追加：

```python
def test_tavily_api_key_defaults_to_empty():
    with patch.dict(os.environ, {}, clear=True):
        s = Settings()
        assert s.tavily_api_key == ""


def test_tavily_api_key_reads_from_env():
    with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test123"}):
        s = Settings()
        assert s.tavily_api_key == "tvly-test123"
```

注意：`tests/test_config.py` 中可能已有 `from unittest.mock import patch` 和 `import os` 以及 `from biteme.config import Settings`。如果没有，需要在文件顶部添加。

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n agent pytest tests/test_config.py::test_tavily_api_key_defaults_to_empty -v
```

Expected: FAIL（`Settings` 没有 `tavily_api_key` 属性）

- [ ] **Step 3: 修改 `biteme/config.py`**

在 `Settings.__init__` 中，`self.github_token` 行之后添加：

```python
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
```

- [ ] **Step 4: 运行测试确认通过**

```bash
conda run -n agent pytest tests/test_config.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add biteme/config.py tests/test_config.py
git commit -m "feat: add TAVILY_API_KEY to Settings"
```

---

### Task 4: 重写 answerer prompt

**Files:**
- Modify: `biteme/graph/prompts.py`
- Test: `tests/test_graph_nodes.py`

- [ ] **Step 1: 写测试**

在 `tests/test_graph_nodes.py` 末尾追加：

```python
def test_answerer_prompt_no_context_placeholder():
    """Answerer prompts should not contain {context} — context is now passed via HumanMessage."""
    for mode in ("learn", "interview"):
        prompts = get_prompts(mode)
        assert "{context}" not in prompts["answerer"], (
            f"{mode} answerer prompt still contains '{{context}}' placeholder"
        )
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_answerer_prompt_no_context_placeholder -v
```

Expected: FAIL（当前 prompt 包含 `{context}`）

- [ ] **Step 3: 修改 `biteme/graph/prompts.py`**

将 `LEARN_ANSWERER` 替换为：

```python
LEARN_ANSWERER = """\
你是内容专家。基于提供的检索片段详细回答问题。
如果检索片段不足以完整回答，使用工具获取更多源码或文件内容来补充。
回答中引用文件路径时请标注出来。
如果工具调用后仍无法获取足够信息，请诚实说明。
"""
```

将 `INTERVIEW_ANSWERER` 替换为：

```python
INTERVIEW_ANSWERER = """\
你是一位技术面试候选人。给出简洁、专业的技术回答。
如果提供的参考信息不够充分，使用工具查阅更多源码或相关资料来支撑你的回答。
不要直接照抄参考内容，用自己的语言组织回答。
"""
```

- [ ] **Step 4: 运行测试确认通过**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_answerer_prompt_no_context_placeholder -v
```

Expected: PASS

- [ ] **Step 5: 运行所有已有 prompt 测试**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_get_prompts_learn_has_planner tests/test_graph_nodes.py::test_get_prompts_interview_has_planner -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add biteme/graph/prompts.py tests/test_graph_nodes.py
git commit -m "feat: rewrite answerer prompts for ReAct (remove {context} placeholder, add tool guidance)"
```

---

### Task 5: 将 `answerer_node` AI 分支改为 ReAct

**Files:**
- Modify: `biteme/graph/nodes.py:123-181`
- Modify: `tests/test_graph_nodes.py`

- [ ] **Step 1: 写测试**

在 `tests/test_graph_nodes.py` 中，找到 `test_answerer_node_always_retrieves`，将其替换为以下两个测试：

```python
def test_answerer_node_always_retrieves(tmp_path):
    """answerer_node must call provider.retrieve() regardless of ReAct."""
    state = make_state(
        source_path=str(tmp_path),
        current_speaker="answerer",
        messages=[{"speaker": "questioner", "content": "What does foo do?", "retrieved_chunks": []}],
    )
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_react_agent = MagicMock()
    mock_react_agent.invoke.return_value = {
        "messages": [MagicMock(content="foo returns None.")]
    }

    with patch("biteme.graph.nodes.create_react_agent", return_value=mock_react_agent), \
         patch("biteme.graph.nodes.ChatOpenAI"), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.retrieve.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        result = answerer_node(state)

    mock_provider.retrieve.assert_called_once()
    assert result["current_speaker"] == "questioner"
    assert result["messages"][-1]["speaker"] == "answerer"
    assert len(result["messages"][-1]["retrieved_chunks"]) > 0


def test_answerer_node_calls_react_agent(tmp_path):
    """AI branch should use create_react_agent instead of direct llm.invoke()."""
    state = make_state(
        source_path=str(tmp_path),
        current_speaker="answerer",
        messages=[{"speaker": "questioner", "content": "Explain the architecture.", "retrieved_chunks": []}],
    )
    (tmp_path / "a.py").write_text("class App: pass")

    mock_react_agent = MagicMock()
    mock_react_agent.invoke.return_value = {
        "messages": [MagicMock(content="The architecture is based on...")]
    }

    with patch("biteme.graph.nodes.create_react_agent", return_value=mock_react_agent) as mock_create, \
         patch("biteme.graph.nodes.ChatOpenAI") as mock_chat, \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.retrieve.return_value = ["class App: pass"]
        mock_factory.return_value = mock_provider
        result = answerer_node(state)

    mock_create.assert_called_once()
    mock_react_agent.invoke.assert_called_once()
    call_kwargs = mock_react_agent.invoke.call_args
    assert call_kwargs[1]["recursion_limit"] == 12
    assert result["messages"][-1]["content"] == "The architecture is based on..."
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_answerer_node_calls_react_agent -v
```

Expected: FAIL（`answerer_node` 还没用 `create_react_agent`）

- [ ] **Step 3: 修改 `biteme/graph/nodes.py`**

在文件顶部 import 区域添加：

```python
from langgraph.prebuilt import create_react_agent
from ..tools import READONLY_TOOLS
```

将 `answerer_node` 函数的 `else` 分支（第 162-175 行）替换为：

```python
    else:
        prompts = get_prompts(state["mode"])
        context_text = "\n\n---\n\n".join(chunks[:5])

        history = "\n".join(
            f"[{t['speaker']}]: {t['content']}" for t in state["messages"][-6:]
        )
        llm = ChatOpenAI(model=settings.openai_model, temperature=0.3)
        react_agent = create_react_agent(
            model=llm,
            tools=READONLY_TOOLS,
            prompt=prompts["answerer"],
        )
        result = react_agent.invoke(
            {"messages": [HumanMessage(
                content=(
                    f"检索到的相关内容：\n{context_text}"
                    f"\n\n对话历史：\n{history}"
                    f"\n\n请回答最后那个问题。"
                )
            )]},
            recursion_limit=12,
        )
        final_answer = result["messages"][-1].content
        turn = {"speaker": "answerer", "content": final_answer, "retrieved_chunks": chunks}
```

- [ ] **Step 4: 运行全部 answerer 测试**

```bash
conda run -n agent pytest tests/test_graph_nodes.py::test_answerer_node_always_retrieves tests/test_graph_nodes.py::test_answerer_node_calls_react_agent -v
```

Expected: 2 passed

- [ ] **Step 5: 运行全部测试**

```bash
conda run -n agent pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add biteme/graph/nodes.py tests/test_graph_nodes.py
git commit -m "feat: convert answerer AI branch to ReAct agent with tool-calling capability"
```

---

### Task 6: 端到端冒烟验证

- [ ] **Step 1: 确认所有测试通过**

```bash
conda run -n agent pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 2: 确认 import 链无报错**

```bash
conda run -n agent python -c "from biteme.graph.nodes import answerer_node; from biteme.tools import READONLY_TOOLS, ALL_TOOLS; print(f'ALL_TOOLS={len(ALL_TOOLS)}, READONLY_TOOLS={len(READONLY_TOOLS)}')"
```

Expected: `ALL_TOOLS=8, READONLY_TOOLS=7`

- [ ] **Step 3: Commit（如有遗漏修复）**

```bash
git status
```

如果有未提交的修复，执行：

```bash
git add -A && git commit -m "fix: address issues found during smoke test"
```
