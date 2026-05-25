# Answerer ReAct 改造设计文档

**日期**: 2026-05-25
**状态**: 待实现

---

## 背景

BiteMe 当前的 `answerer_node` 采用单次 LLM 调用模式：检索上下文 → 拼入 prompt → 调用一次 LLM → 返回回答。这种方式无法在回答过程中主动补充信息。

本次改造将 AI 回答分支改为 ReAct（Reasoning + Acting）模式，让 answerer 在发现初始检索内容不足时，能自主调用工具获取更多代码/文件/网络信息来辅助回答。

---

## 设计决策

### 为什么选择 `create_react_agent` 子图方案

评估了三种方案：

1. **LangGraph ToolNode 平铺在主图**：answerer 拆成两个主图节点（LLM + ToolNode），通过条件边路由。单节点可行，但如果后续 questioner、planner 也要 ReAct 化，主图会膨胀到 6+ 节点，路由混乱。
2. **`create_react_agent` 子图（采用）**：用 LangGraph 预构建的 ReAct agent 作为子图嵌入节点内部。主图结构不变，后续其他节点可复用同一模式。
3. **纯 Python 循环**：在函数内手写 while 循环。绕过 LangGraph 机制，丧失可观测性。

选择方案 2，兼顾当前需求和后续扩展性。

### 设计约束

- **只读工具**：answerer 只使用读取和搜索类工具，不提供写文件能力。
- **HITL 分支不变**：`"answerer" in hitl_flags` 时仍走人类回答流程，ReAct 只影响 AI 回答分支。
- **主图结构不变**：`planner → questioner → answerer → (继续/结束)` 流程和 `graph.py` 无需修改。
- **State 不变**：`SessionState` 不需要新增字段。

---

## 可用工具

answerer ReAct agent 可调用以下 7 个只读工具：

| 工具 | 来源 | 用途 |
|---|---|---|
| `read_file` | `langchain-community` | 读取本地文件内容 |
| `search_files_by_name` | `langchain-community` | 按文件名搜索本地文件 |
| `search_files_by_content` | 自定义 | 按内容搜索本地文件（类 grep） |
| `github_list_tree` | 自定义 | 浏览 GitHub 仓库目录结构 |
| `github_read_file` | 自定义 | 读取 GitHub 仓库文件内容 |
| `github_search_code` | 自定义 | 在 GitHub 仓库中搜索代码 |
| `tavily_search` | `tavily-python` | 联网搜索，获取网络信息 |

排除 `write_file`，answerer 不需要写文件能力。

---

## ReAct Agent 构建

```python
from langgraph.prebuilt import create_react_agent
from biteme.tools import READONLY_TOOLS

react_agent = create_react_agent(
    model=llm,
    tools=READONLY_TOOLS,
    prompt=answerer_system_prompt,
)

result = react_agent.invoke(
    {"messages": [HumanMessage(content=...)]},
    {"recursion_limit": 12},
)
final_answer = result["messages"][-1].content
```

### 迭代上限

`recursion_limit=12`。ReAct 内部每轮工具调用消耗 2 步（LLM + ToolNode），5 轮 = 10 步，加上首次 LLM 调用 = 11 步，12 为安全余量。超过上限后 agent 自动停止并返回当前最后一条消息。

### 输入构造

ReAct agent 接收一条 HumanMessage，包含：

1. 初始检索到的上下文（与当前逻辑一致，仍通过 `provider.retrieve()` 获取）
2. 最近 6 轮对话历史
3. 回答指令

初始 context 让 LLM 先看到已有信息，只在不够时才主动调用工具补充。

### 输出提取

从 `result["messages"][-1].content` 提取最终文本回答，包装成 `Turn(speaker="answerer", content=..., retrieved_chunks=chunks)`，对主图完全透明。

---

## Prompt 改动

当前 prompt 将检索 context 通过 `{context}` 占位符嵌入 system prompt。改造后拆为两部分：

### System Prompt（固定，传给 `create_react_agent` 的 `prompt` 参数）

**Learn 模式：**

```
你是内容专家。基于提供的检索片段详细回答问题。
如果检索片段不足以完整回答，使用工具获取更多源码或文件内容来补充。
回答中引用文件路径时请标注出来。
如果工具调用后仍无法获取足够信息，请诚实说明。
```

**Interview 模式：**

```
你是一位技术面试候选人。给出简洁、专业的技术回答。
如果提供的参考信息不够充分，使用工具查阅更多源码或相关资料来支撑你的回答。
不要直接照抄参考内容，用自己的语言组织回答。
```

### HumanMessage（动态，每次调用时构造）

```
检索到的相关内容：
{context_text}

对话历史：
{history}

请回答最后那个问题。
```

---

## 文件改动清单

### 新增文件

**`biteme/tools/web.py`**

封装 Tavily 搜索工具：

```python
from langchain_tavily import TavilySearch

tavily_search = TavilySearch(max_results=3)
```

### 修改文件

**`biteme/tools/__init__.py`**

- 导入 `tavily_search`
- 将其加入 `ALL_TOOLS`
- 新增 `READONLY_TOOLS` 列表（`ALL_TOOLS` 中排除 `write_file`）

**`biteme/graph/prompts.py`**

- 修改 `LEARN_ANSWERER`：去掉 `{context}` 占位符，改为角色定义 + 工具使用指引
- 修改 `INTERVIEW_ANSWERER`：同上
- `get_prompts()` 接口不变

**`biteme/graph/nodes.py`**

- 新增 import：`create_react_agent`、`READONLY_TOOLS`
- `answerer_node` 的 `else`（AI 回答）分支：用 `create_react_agent` 构建 agent 并调用，替代原来的单次 `llm.invoke()`
- HITL 分支（`if "answerer" in hitl_flags`）完全不动

**`biteme/config.py`**

- 新增 `self.tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")`

### 不需要改的文件

- `biteme/graph/graph.py`（主图结构不变）
- `biteme/graph/state.py`（无新增状态字段）
- `biteme/tools/filesystem.py`（工具实现不变）
- `biteme/tools/github.py`（工具实现不变）

---

## 依赖新增

- `langchain-tavily`：LangChain 官方的 Tavily 集成包（内部依赖 `tavily-python`，无需单独安装）

---

## 不在范围内

- questioner / planner 的 ReAct 化（后续可复用同一模式）
- 写文件类工具的接入
- 工具调用的流式输出展示
- 工具调用结果的持久化存储
