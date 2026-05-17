# Planner Agent 设计文档

**日期**：2026-05-17  
**状态**：已批准，待实现

---

## 概述

在现有的 questioner → answerer 双 Agent 循环之前，固定插入一个 **planner 节点**，在会话开始时根据文档内容和 `max_turns` 生成提问大纲（具体问题列表），展示给用户后自动继续。大纲供后续 questioner 参考使用，不论模式（learn/interview）或 HITL 配置如何，planner 始终执行。

---

## 架构

### 图拓扑

```
[会话开始]
    ↓
 planner          ← 新增，固定入口
    ↓
questioner  ←─────────────────────┐
    ↓                             │
answerer                          │
    ↓                             │
[turn_count < max_turns?] ─ yes ──┘
    ↓ no
  END
```

- `planner` 是新的图入口（`set_entry_point("planner")`）
- `planner → questioner` 为固定无条件边
- Resume 时 LangGraph 从 checkpoint 恢复到上次中断的节点，不重新经过 planner

---

## State 变更

在 `SessionState`（`biteme/graph/state.py`）新增一个字段：

```python
outline: list[str]   # planner 生成的提问大纲，空列表表示尚未生成
```

- 初始值为 `[]`（在 `cli.py` 的 `initial_state` 中设置）
- planner 执行后填入 `max_turns + 2` 个问题字符串
- 随 LangGraph checkpoint 持久化，resume 时自动恢复

---

## Planner 节点（`biteme/graph/nodes.py`）

### 职责

1. 调用 `context provider` 的 `get_overview()` 获取文档摘要
2. 调用 LLM，根据文档摘要、`max_turns`、`mode` 生成 `max_turns + 2` 个问题
3. 将问题列表存入 `state["outline"]`
4. 解析 LLM 输出：按行分割，过滤掉空行，用正则去掉行首的 `1. ` / `2. ` 等编号前缀，得到纯问题字符串列表
5. 用 Rich Panel 在终端展示大纲
6. 无任何 interrupt，自动透传控制权给 questioner

### Context 策略

只使用 `get_overview()`，不做 RAG 检索，保持 planner 轻量快速。

### Prompt

`prompts.py` 中新增 `LEARN_PLANNER` 和 `INTERVIEW_PLANNER`，结构与现有 prompt 一致：

**LEARN_PLANNER（learn 模式）**
```
你是一位学习规划者。根据以下文档摘要，为一个将要学习该文档的人
规划 {n} 个由浅入深的学习问题。
要求：
- 覆盖文档的主要知识点
- 问题之间有逻辑递进关系
- 每个问题独立成句，只输出问题本身
- 按编号列出，格式：1. xxx  2. xxx ...
```

**INTERVIEW_PLANNER（interview 模式）**
```
你是一位技术面试规划者。根据以下文档摘要，为一场技术面试
规划 {n} 个考察问题，从基础到深度递进。
要求：
- 严格基于文档内容出题
- 覆盖核心技术点
- 每个问题独立成句，只输出问题本身
- 按编号列出，格式：1. xxx  2. xxx ...
```

`get_prompts()` 函数扩展，增加 `"planner"` key。

### 展示格式

使用 Rich Panel 展示，标题为 `"提问大纲"` 或 `"面试大纲"`，内容为编号列表，颜色与现有 CLI 风格一致（青色/黄色）。

---

## Questioner 节点变更（`biteme/graph/nodes.py`）

### AI questioner 模式

在发给 LLM 的 `HumanMessage` 中附加大纲上下文：

```
对话历史：
{history}

提问大纲（供参考，请结合对话历史灵活使用，不必按顺序）：
{outline_text}

参考内容摘要：
{context_text}

请提出下一个问题。
```

`outline_text` 由 `outline` 列表格式化为编号字符串，`outline` 为空时不附加该段落。

### HITL questioner 模式

`interrupt()` 的提示文字改为：

```
>>> [提问者] 建议问题（第 {turn_count+1} 轮）：{suggested}
（直接回车使用建议问题，或输入新问题）
```

其中 `suggested = outline[turn_count] if turn_count < len(outline) else ""`。

`cli.py` 的 `_run_graph()` 处理 interrupt 时：若用户输入为空字符串，则使用 `interrupt_obj.value` 中解析出的建议问题原文作为实际输入。

> **注意**：`turn_count` 在 questioner 节点末尾递增（`+1`），因此取建议问题时使用递增**前**的值（即 `state["turn_count"]`）。

---

## CLI 变更（`biteme/cli.py`）

### `initial_state` 新增字段

```python
initial_state: SessionState = {
    ...
    "outline": [],   # 新增
}
```

### HITL 空回车处理

在 `_run_graph()` 的 interrupt 处理分支中：

```python
user_input = typer.prompt("", prompt_suffix="> ")
if not user_input.strip():
    # 从 interrupt 提示文字中提取建议问题
    user_input = _extract_suggestion(prompt_text)
pending = Command(resume=user_input)
```

`_extract_suggestion()` 是一个简单的字符串解析函数，从提示文字中提取建议问题部分。

---

## 文件改动清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `biteme/graph/state.py` | 修改 | 新增 `outline: list[str]` |
| `biteme/graph/nodes.py` | 修改 | 新增 `planner_node`；修改 `questioner_node` |
| `biteme/graph/prompts.py` | 修改 | 新增 `LEARN_PLANNER`、`INTERVIEW_PLANNER`；扩展 `get_prompts()` |
| `biteme/graph/graph.py` | 修改 | 新增 planner 节点和边；更改入口点 |
| `biteme/cli.py` | 修改 | `initial_state` 增加 `outline: []`；HITL 空回车逻辑 |

---

## 测试要点

- planner 节点在全部 4 种 HITL 组合下均只执行一次
- resume 后 outline 正确从 checkpoint 恢复，不重新生成
- AI questioner 收到带大纲的 prompt
- HITL questioner 空回车时使用建议问题
- `outline` 为空（异常情况）时 questioner 能正常降级运行
