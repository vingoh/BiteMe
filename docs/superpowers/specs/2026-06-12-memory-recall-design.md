# Memory Recall & Question Refine — 设计规格

**日期：** 2026-06-12  
**状态：** 待实现  
**相关文件：** `biteme/graph/memory.py`、`biteme/graph/prompts.py`、`biteme/graph/nodes.py`

---

## 背景

当前 `questioner_node` 每轮提问仅依赖 planner 大纲和上下文检索，完全不感知用户的历史掌握情况。记忆系统（`memory.json`）虽然持续写入每轮答题的评分和评语，但从未在出题侧被利用。

本功能在 `questioner_node` 内部新增两个步骤：**记忆召回（recall）** 和 **问题精炼（refine）**，使出题策略能够主动规避已掌握话题、深挖薄弱环节、并对久未复习的知识点进行间隔重复。

---

## 目标

- questioner 提出的草稿问题，在发出前先经过"记忆召回 → 问题精炼"两步处理
- 召回：从 memory.json 中找出与草稿问题最相关的 ≤3 条记忆
- 精炼：根据召回记忆的分数（avg_score）和最后更新日期（last_update）调整问题方向
- 对两个 mode（learn / interview）均生效
- HITL questioner 模式下，召回的记忆展示给人类参考
- 任何步骤失败均降级为直接使用草稿问题，不中断主流程

---

## 架构

### 节点内部流程

`questioner_node` 保持为单一 LangGraph 节点，内部分三个阶段：

```
questioner_node()
  │
  ├─ [Phase 1] draft_question
  │   LLM questioner 路径：stream_agent(react_agent, ...)   ← 现有逻辑不变
  │   HITL questioner 路径：outline 中的建议问题              ← 现有逻辑不变
  │
  ├─ [Phase 2] recalled = recall_memory(draft_question, memory_data)
  │   LLM structured output → list[RecalledEntry]，≤3 条
  │   memory 为空 / 调用失败 / draft 为空字符串 → recalled = []
  │
  ├─ [Phase 3] 
  │   LLM questioner 路径：
  │     if recalled: final_question = refine_question(draft, recalled, memory_data)
  │     else:        final_question = draft
  │   HITL questioner 路径：
  │     if recalled: console.print(Panel(recalled))   ← 展示记忆参考
  │     human_text = interrupt(prompt_msg)             ← 人类自行决策
  │
  └─ return Turn{speaker, content=final_question / human_text}
```

### 图结构

`graph.py` 和 `state.py` **完全不变**。recalled memories 仅作为 `questioner_node` 内部局部变量，不持久化到 `SessionState`。

---

## 新增数据模型

位于 `biteme/graph/memory.py`：

```python
class RecalledEntry(BaseModel):
    key: str
    relevance_reason: str  # 必须引用 alias 或 comment 中的原文作为依据

class MemoryRecallResult(BaseModel):
    recalled: list[RecalledEntry]  # max 3
```

`recall_memory()` 内部拼装完整的召回条目（合并 `RecalledEntry` 与 `MemoryEntry` 中的 `avg_score`、`last_update`），供 `refine_question()` 使用。

---

## 新增函数

### `recall_memory(draft_question, memory_data, model=None) -> list[RecalledEntry]`

- 输入：draft question（str）、完整 MemoryFile
- 若 `memory_data["entries"]` 为空，立即返回 `[]`
- 调用 LLM（`with_structured_output(MemoryRecallResult)`）
- 解析失败 → 返回 `[]`，记 warning log
- 过滤掉 recalled 中 key 不在 memory entries 的条目

### `refine_question(draft_question, recalled, memory_data, model=None) -> str`

- 输入：draft question（str）、`list[RecalledEntry]`、完整 MemoryFile（用于读取 avg_score / last_update / comments）
- 拼装 recalled_entries 时包含：key、avg_score、last_update、relevance_reason、comments.strength（最近 3 条）、comments.weakness（最近 3 条）
- 调用简单 LLM（非 ReAct，单次 `.invoke()`）
- 返回精炼后的问题字符串
- 调用失败或返回空字符串 → 返回 draft_question（降级）

---

## Prompts

### `MEMORY_RECALL_PROMPT`

```
你是一个记忆检索助手。根据【草稿问题】，从【已有记忆】中选出相关性最高的至多 3 条。

## 草稿问题
{draft_question}

## 已有记忆
{memory_entries}
（每条包含：key、aliases、comments.strength、comments.weakness）

## 判断标准

### 标准一：Alias 直接命中
entry 的 aliases 中有与草稿问题核心概念相同或高度重叠的说法，即便措辞不同。

例子（草稿问题："BPE tokenization 是如何处理 OOV 词的？"）：
- aliases 含 ["BPE", "Byte Pair Encoding", "子词分词"] → ✅ 直接命中
- aliases 含 ["tokenizer", "分词器"] → ⚠️ 需结合 comments 判断，单靠 alias 不够
- aliases 含 ["Transformer 架构", "注意力机制"] → ❌ 不相关

### 标准二：Comments 话题关联
即使 alias 未直接命中，entry 的 strength 或 weakness comments 中提到了草稿问题涉及的
具体概念、机制、或相关术语，也视为相关。

例子（草稿问题："梯度裁剪（gradient clipping）在训练中的作用？"）：
- weakness 含 "用户没有解释梯度爆炸的触发条件" → ✅ 梯度爆炸是梯度裁剪要解决的问题，高度关联
- strength 含 "用户正确描述了 clip_grad_norm 的用法" → ✅ 直接提到相关 API
- comments 含 "用户对反向传播过程描述完整" → ⚠️ 上下游相关但不直接，仅在无更好候选时考虑
- comments 含 "用户对学习率调度理解到位" → ❌ 同属优化领域但话题不同

### 不算相关的情况
- 仅因同属一个大领域（例如都是"深度学习"、"Python"、"NLP"）→ ❌
- key 名称中有相似词但 aliases 和 comments 均无交集 → ❌
- 相关性纯靠猜测/推断，没有 alias 或 comment 中的直接证据 → ❌

## 输出
只输出 JSON，不含 markdown 围栏：
{"recalled": [{"key": "...", "relevance_reason": "..."}, ...]}
relevance_reason 必须引用具体 alias 或 comment 中的文字作为依据。
最多 3 条，完全不相关时输出 {"recalled": []}。
```

### `MEMORY_REFINE_PROMPT`

```
你是一位提问优化助手。根据用户的历史掌握情况，对【草稿问题】进行调整，
使问题更有针对性地帮助用户查漏补缺。

## 草稿问题
{draft_question}

## 相关历史记忆（按相关性排序）
{recalled_entries}
（每条包含：key、avg_score(0-10)、last_update、relevance_reason、
  comments.strength（用户在此话题上的具体优点）、
  comments.weakness（用户在此话题上的具体错误或遗漏））

## 调整规则

### 基于分数与日期
- avg_score ≥ 7 且 last_update 在 14 天内：用户已近期掌握，转向相邻话题或追问更深层细节
- avg_score ≤ 4：薄弱环节，保持问题方向，可适当简化难度从基础考起
- last_update 超过 14 天（无论分数）：久未复习，保持或强化原方向
- 若所有召回记忆均为高分且近期：在同领域内换一个新角度提问

### 基于 Comments（优先级高于分数规则）
- weakness 中有具体错误或遗漏 → 针对该错误/遗漏设计问题，让用户有机会弥补
  例：weakness "没有解释梯度爆炸的触发条件" → 问题可聚焦在梯度爆炸的触发机制上
- strength 中有某项能力已熟练掌握 → 避免重复考察该项，转向 weakness 或更深层问题
  例：strength "正确描述了 clip_grad_norm 的用法" → 不再考用法，改考原理或边界情况
- weakness 和 strength 均为空 → 退回分数/日期规则

## 输出
只输出最终问题本身，不含任何前缀或解释。
```

---

## 错误处理与降级

| 场景 | 处理方式 |
|---|---|
| `memory.json` 不存在或为空 | 跳过 recall + refine，直接用 draft |
| HITL 模式下 outline 为空（无建议问题） | draft 为空字符串，跳过 recall，静默 |
| recall LLM 调用失败 / parse 失败 | `recalled = []`，跳过 refine，用 draft，记 warning |
| recall 返回 `{"recalled": []}` | 跳过 refine，直接用 draft |
| recalled key 不在 memory entries 中 | 过滤掉该条目，用剩余有效条目继续 |
| refine LLM 调用失败 | 返回 draft，记 warning |
| refine 返回空字符串 | 返回 draft |

所有失败记 `logger.warning`，不抛异常，不中断图执行。

---

## HITL Questioner 展示

当人类担任 questioner 时，在 `interrupt()` 之前展示召回的记忆：

```
╭─────────────── 相关记忆参考 ───────────────╮
│  • bpe_tokenization  avg_score=3.5  最近: 2026-05-20  │
│    相关依据: aliases 含 "BPE"                         │
│                                                        │
│  • gradient_clipping  avg_score=8.0  最近: 2026-06-10 │
│    相关依据: weakness 提到 "梯度爆炸触发条件"          │
╰────────────────────────────────────────────╯
建议问题（第 3 轮）：...
（直接回车使用建议问题，或输入新问题）
```

- `recalled` 为空时静默，不显示 Panel
- 人类有完全控制权，记忆仅作参考

---

## 文件改动范围

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `biteme/graph/memory.py` | 新增 | `RecalledEntry`、`MemoryRecallResult`、`recall_memory()`、`refine_question()` |
| `biteme/graph/prompts.py` | 新增 | `MEMORY_RECALL_PROMPT`、`MEMORY_REFINE_PROMPT`；`get_prompts()` 新增 `"recall"`、`"refine"` key |
| `biteme/graph/nodes.py` | 修改 | `questioner_node()` 内插入 Phase 2（recall）和 Phase 3（refine / HITL 展示） |
| `biteme/graph/graph.py` | 不变 | — |
| `biteme/graph/state.py` | 不变 | — |

---

## 分数与日期阈值（可配置）

| 参数 | 默认值 | 含义 |
|---|---|---|
| `SCORE_HIGH` | 7 | avg_score ≥ 此值视为已掌握 |
| `SCORE_LOW` | 4 | avg_score ≤ 此值视为薄弱 |
| `STALE_DAYS` | 14 | last_update 距今超过此天数视为久未复习 |

初期硬编码为常量，后续可移入 `config.py`。
