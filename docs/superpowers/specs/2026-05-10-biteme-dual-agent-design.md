# BiteMe 双 Agent 问答系统设计文档

**日期**：2026-05-10  
**状态**：已批准，待实现

---

## 概述

BiteMe 是一个双 Agent 问答系统，核心由两个角色构成：**提问者（Questioner）** 和 **回答者（Answerer）**。系统以给定的内容（代码仓库、文档、简历等）为知识来源，两个 Agent 围绕内容进行多轮对话。人类用户可以作为纯观察者旁观，也可以在任意一侧接管为人工输入。

典型使用场景：

- **仓库导读**：Questioner 提出探索性问题，Answerer 基于代码逐步解释仓库结构与实现
- **模拟面试**：Questioner 扮演面试官出技术题，Answerer（或人类）作答并获得评价

---

## 技术栈

| 层次 | 选型 |
|------|------|
| 编排 + HITL | LangGraph |
| LLM | OpenAI API（GPT-4o） |
| RAG 管线 | LangChain loaders + splitters + retriever |
| 向量库 | LanceDB（本地嵌入式，无需额外服务） |
| Embedding | OpenAI text-embedding-3-small |
| CLI | Typer + Rich |
| 会话持久化 | LangGraph SQLite Checkpoint |

---

## 架构

### 整体数据流

```
CLI (Typer + Rich)
    │
    ├─► Indexing Pipeline（离线，大内容时预先运行）
    │       └─► LanceDB（~/.biteme/indexes/）
    │
    └─► LangGraph Session（在线对话图）
            ├── questioner_node（内含 interrupt() HITL 中断点）
            ├── answerer_node（内含 interrupt() HITL 中断点，每轮必须经过 ContextProvider）
            └── SQLite Checkpoint（~/.biteme/sessions/）
```

### ContextProvider 抽象

两种策略暴露同一接口，图里的节点对底层策略无感知：

```python
class ContextProvider(ABC):
    def retrieve(self, query: str) -> list[str]: ...
```

- **DirectProvider**：读取文件全文，直接以字符串列表返回（适合单文件、简历等小内容）
- **RAGProvider**：从 LanceDB 检索 top-k 相关 chunk，返回片段列表（适合大型仓库、多文档集合）

**自动选择逻辑（factory）**：

```
默认 auto：估算内容 token 数
    ≤ 100K token  →  DirectProvider
    > 100K token  →  RAGProvider（需先运行 biteme index）

用户可用 --strategy direct|rag 强制覆盖
```

---

### LangGraph 状态

```python
class Turn(TypedDict):
    speaker: str           # "questioner" | "answerer" | "human"
    content: str
    retrieved_chunks: list[str]   # answerer 轮携带检索结果，其余为空列表

class SessionState(TypedDict):
    mode: Literal["learn", "interview"]
    messages: list[Turn]
    current_speaker: str           # "questioner" | "answerer"
    hitl_flags: list[str]          # 哪些角色由人类控制，用 list 保证 SQLite 可序列化
    turn_count: int
    max_turns: int
    context_strategy: str          # "auto" | "direct" | "rag"
    source_path: str               # 内容来源路径
```

`hitl_flags` 在会话启动时由 CLI 写入，运行过程中不变：

```python
hitl_flags = []                                     # 纯观察者，两侧均为 AI
hitl_flags = ["questioner"]                         # 仅提问侧由人类控制
hitl_flags = ["answerer"]                           # 仅回答侧由人类控制
hitl_flags = ["questioner", "answerer"]             # 两侧均为人类
```

---

### 节点逻辑

**questioner_node**：

1. 若 `"questioner" in hitl_flags` → `interrupt()`，CLI 接收人类输入
2. 否则：
   - 从 `state["messages"]` 读取完整对话历史（已问过哪些问题、收到了哪些回答）
   - **可选**：以当前话题为 query，调用 `ContextProvider.retrieve(query)` 从知识库取几段参考内容，辅助生成更贴近内容的问题；query 字符串本身由对话历史推导而来
   - 将对话历史 + 可选的检索片段一起传给 LLM，生成下一个问题
3. 追加 Turn，切换 `current_speaker → "answerer"`

**answerer_node**：

1. 从 `state["messages"]` 读取对话历史，取出最后一条问题的文本（`messages[-1]["content"]`）
2. 以该问题文本为 query，调用 `ContextProvider.retrieve(query)` 从知识库获取相关片段（**必选，不可跳过**）
3. 若 `"answerer" in hitl_flags` → `interrupt()`，CLI 展示检索片段后接收人类输入
4. 否则：将对话历史 + 检索片段一起注入 prompt，调用 LLM 生成回答
5. 追加 Turn（携带 `retrieved_chunks`），切换 `current_speaker → "questioner"`

> **说明**：`state["messages"]` 是对话历史的唯一来源；`ContextProvider.retrieve(query)` 仅用于从外部知识库取相关内容，两者职责不重叠。

**终止条件**：`turn_count >= max_turns` 时图结束。

---

### 两种模式的 System Prompt 差异

两种模式共用同一张图，仅注入的 system prompt 不同：

| | learn 模式 | interview 模式 |
|---|---|---|
| Questioner | 好奇的学习者，针对内容提出探索性、递进式问题 | 技术面试官，提出有深度的技术问题并在每轮末简短评价上一轮回答 |
| Answerer | 内容专家，基于检索片段详细解释，引用来源路径 | 面试候选人，给出简洁专业的技术回答 |

---

## CLI 设计

```
biteme index <source>
    # 预建 RAG 索引，适用于大型代码仓或多文档集合
    # source 可以是本地目录路径或单个文件路径

biteme run <source>
    --mode learn|interview         # 默认 learn
    --hitl none|questioner|answerer|both   # 默认 none
    --strategy auto|direct|rag     # 默认 auto
    --turns N                      # 最大对话轮数，默认 10

biteme resume <session-id>
    # 恢复上次中断（HITL 暂停或 Ctrl+C 退出）的会话

biteme list
    # 列出历史会话（session-id、source、mode、时间、状态）
```

---

## 目录结构

```
biteme/
├── __init__.py
├── cli.py                    # CLI 入口（Typer app）
├── config.py                 # 环境变量与配置（OPENAI_API_KEY 等）
├── context/
│   ├── __init__.py
│   ├── base.py               # ContextProvider 抽象基类
│   ├── direct.py             # DirectProvider
│   ├── rag.py                # RAGProvider（LanceDB）
│   └── factory.py            # 自动选择策略（auto/direct/rag）
├── graph/
│   ├── __init__.py
│   ├── state.py              # SessionState + Turn TypedDict
│   ├── prompts.py            # 各模式 system prompt 模板
│   ├── nodes.py              # questioner_node, answerer_node
│   └── graph.py              # 构建 LangGraph StateGraph
├── session/
│   ├── __init__.py
│   └── manager.py            # 会话创建、列表、resume（SQLite checkpoint）
└── indexing/
    ├── __init__.py
    └── pipeline.py           # 建索引管线（loader → splitter → embed → LanceDB）

tests/
├── test_context_direct.py
├── test_context_rag.py
├── test_context_factory.py
├── test_graph_nodes.py
├── test_session_manager.py
└── test_cli.py

pyproject.toml
README.md
```

---

## 错误处理

| 场景 | 处理方式 |
|------|----------|
| OpenAI API 调用失败 | 捕获异常，Rich 打印错误详情，提示用户重试或退出 |
| 文件/目录不存在 | CLI 层校验，立即报错，不进入图 |
| auto 策略判断需要 RAG 但索引未建 | 提示用户先运行 `biteme index <source>`，退出 |
| HITL 中断后用户 Ctrl+C | 会话通过 checkpoint 保存，可用 `biteme resume` 继续 |

---

## 会话持久化

- 使用 LangGraph 内置 `SqliteSaver` checkpoint
- 数据存储路径：`~/.biteme/sessions/<session-id>.db`
- 每轮对话结束后自动保存状态
- `biteme list` 从所有 `.db` 文件读取元数据展示
