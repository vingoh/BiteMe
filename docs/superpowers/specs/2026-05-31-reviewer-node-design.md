# Reviewer Node Design

**Date:** 2026-05-31  
**Status:** Approved

## Overview

Add a `reviewer_node` to the LangGraph pipeline that, after a user answers a question in HITL mode, compares the user's answer against the LLM-generated reference answer, extracts 1–3 keywords, and scores each keyword (0–10) to surface the user's weak and strong areas over time.

## Goals

- Score each user answer on a per-keyword basis (no overall score)
- Accumulate results in state across all turns
- Display keywords and scores immediately after each answer
- Only trigger in HITL mode (when `"answerer" in hitl_flags`)

## State Changes (`state.py`)

Two new TypedDicts and one new field on `SessionState`:

```python
class KeywordScore(TypedDict):
    keyword: str    # e.g. "向量检索"
    score: int      # 0–10; measures how well the user covered this concept

# SessionState gets:
review_history: list[list[KeywordScore]]
# review_history[i] = keyword scores for turn i
# index in the list encodes which turn; no wrapper needed
```

`review_history` is initialized to `[]` and appended once per reviewer run.

## Graph Topology (`graph.py`)

Current flow:
```
planner → questioner → answerer → [_should_continue] → questioner / END
```

New flow:
```
planner → questioner → answerer → [_after_answerer] → reviewer / questioner / END
                                   reviewer → [_should_continue] → questioner / END
```

### Routing functions

**`_after_answerer(state)`** — replaces the old `_should_continue` on `answerer`:
- `"answerer" in hitl_flags` → `"reviewer"`
- `turn_count >= max_turns` → `END`
- else → `"questioner"`

**`_should_continue(state)`** — unchanged, now only hangs off `reviewer`:
- `turn_count >= max_turns` → `END`
- else → `"questioner"`

Both paths (HITL and non-HITL) correctly reach `END` when `max_turns` is exhausted.

## Reviewer Node (`nodes.py`)

### Inputs (read from state)

| Field | Source |
|---|---|
| Question | `state["messages"][-2]["content"]` |
| User answer | `state["messages"][-1]["content"]` |
| LLM reference answer | `state["llm_reference_answer"]` |

### LLM call

- Model: `settings.openai_model`, temperature `0.1`
- Plain `ChatOpenAI` (no ReAct agent, no tools — pure structured output)
- Output: JSON only, no extra text

### Prompt (`prompts.py`)

New constant `REVIEWER` (shared between learn and interview modes):

```
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
```

### JSON parsing

Use `json.loads()` on the LLM response. If parsing fails, return an empty list for that turn (no crash).

### Display

Use `rich.Panel` with color-coded scores:
- `score >= 7` → green
- `4 <= score < 7` → yellow
- `score < 4` → red

Example output:
```
╭─── 本轮评审 ───╮
│ 向量检索   8    │  (green)
│ reranker   3    │  (red)
╰─────────────────╯
```

### State return

```python
return {
    "review_history": state["review_history"] + [keywords_list]
}
```

## Files Changed

| File | Change |
|---|---|
| `biteme/graph/state.py` | Add `KeywordScore`, update `SessionState` |
| `biteme/graph/prompts.py` | Add `REVIEWER` constant; expose via `get_prompts()` |
| `biteme/graph/nodes.py` | Add `reviewer_node` function |
| `biteme/graph/graph.py` | Add node, replace routing functions, update edges |

## Out of Scope

- Final session summary (aggregated view of all turns' keywords) — future work
- Reviewer in non-HITL mode
- Persisting review history to disk outside of LangGraph checkpointer
