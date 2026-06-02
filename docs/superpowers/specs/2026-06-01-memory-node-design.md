# Memory Node Design

## Overview

Add a `memory_node` to the BiteMe LangGraph pipeline. The node runs at session end, reads `review_history` from the current session's `SessionState`, merges it with the user's persistent memory file via a single LLM call, and writes the result back to disk. The goal is to accumulate per-keyword mastery scores across sessions so that future nodes (e.g., `questioner_node`) can personalize their behavior based on the user's learning history.

This spec covers only the `memory_node` implementation. Questioner integration is out of scope.

---

## Storage Format

**File path:** `~/.biteme/review_memory.json` (single file, single user; future multi-user extension: `~/.biteme/review_memory/<user_id>.json`)

**Format:** a flat JSON array of keyword entries at the top level.

```json
[
  {
    "keyword": "缩放点积注意力",
    "scores": [6, 8, 9],
    "avg_score": 7.67
  },
  {
    "keyword": "梯度消失",
    "scores": [3],
    "avg_score": 3.0
  }
]
```

Field semantics:

| Field | Type | Description |
|---|---|---|
| `keyword` | string | Canonical name for the concept, chosen by LLM when merging synonyms |
| `scores` | list[int] | Per-session scores appended in chronological order; if a keyword appears in multiple turns within one session, each turn's score is appended separately |
| `avg_score` | float | Mean of all values in `scores`, rounded to 2 decimal places, computed by code after LLM returns the merged `scores` |

`review_memory_path = biteme_home / "review_memory.json"` is added to `Settings`.

---

## Node Behavior

`memory_node` is a pure side-effect node: it reads from and writes to disk, and returns an empty dict (no `SessionState` mutations).

### Execution steps

1. **Guard:** if `state["review_history"]` is empty, return immediately without any LLM call or file I/O.
2. **Read old memory:** load `~/.biteme/review_memory.json`; if the file does not exist, treat as `[]`.
3. **Call LLM:** pass old memory JSON + current session's `review_history` to the LLM with the memory merge prompt (see Prompt Design). Max retries: **3**.
4. **Validate output:** parse JSON, check structure (see Validation Rules).
   - On failure: send the raw output and error message back to the LLM for correction. Repeat up to 3 total attempts.
   - After 3 failures: log a warning, skip file write, return normally.
5. **Compute `avg_score`:** after successful validation, code iterates over each entry and sets `avg_score = round(sum(scores) / len(scores), 2)`.
6. **Write file:** atomically overwrite `~/.biteme/review_memory.json` with the result.
7. **Return `{}`.**
### Graph wiring change

All paths that currently lead to `END` are redirected to `memory_node`; `memory_node` is then connected to `END`. Specifically:

- `_after_answerer`: replace `END` branches with `"memory"`
- `_should_continue`: replace `END` with `"memory"`

---

## Prompt Design

Added to `prompts.py` as `MEMORY_MERGE`:

```
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
```

LLM temperature: `0.0` (deterministic output preferred for structured data).

---

## Validation Rules

After parsing the LLM response, each entry must satisfy:

- `keyword`: non-empty string
- `scores`: non-empty list of integers, each in `[0, 10]`

`avg_score` is not validated from the LLM response; it is computed by code after validation passes. Entries that fail validation are dropped. If the top-level structure is not a JSON array, the entire response is treated as a failure and triggers a retry.

---

## Error Handling

| Situation | Behavior |
|---|---|
| `review_history` is empty | Skip immediately, no LLM call, no file I/O |
| `review_memory.json` unreadable (permissions, corrupt) | Log warning, treat as `[]`, continue |
| LLM returns invalid JSON or wrong structure | Feed output + error back to LLM, retry up to 3 total attempts |
| All 3 retries exhausted | Log warning, skip file write, return normally |
| `review_memory.json` write fails | Log warning, do not raise |

Memory failure never propagates to the session result.

---

## File & Module Changes

| File | Change |
|---|---|
| `biteme/config.py` | Add `review_memory_path = biteme_home / "review_memory.json"` |
| `biteme/graph/prompts.py` | Add `MEMORY_MERGE` prompt string; expose via `get_prompts()` or standalone constant |
| `biteme/graph/nodes.py` | Add `memory_node(state)` function |
| `biteme/graph/graph.py` | Redirect `END` branches to `memory_node`; add `memory_node` to graph |
| `tests/test_memory_node.py` | New test file (see Testing) |

---

## Testing

`tests/test_memory_node.py` covers:

1. **Empty `review_history`**: node returns `{}` immediately, no file written, no LLM called.
2. **First run (no existing file)**: `review_history` with 2 turns correctly produces a new `review_memory.json` with code-computed `avg_score`.
3. **Existing memory + new keywords**: new entries are appended alongside preserved old ones; `avg_score` is recomputed by code.
4. **Semantic merge (mocked LLM)**: "向量检索" and "向量搜索" from different sessions are merged under one canonical name with combined `scores`; `avg_score` is computed by code.
5. **Invalid JSON from LLM → retry succeeds on 2nd attempt**: file is written correctly.
6. **All 3 retries fail**: no file written, no exception raised.
7. **`review_memory.json` write failure**: no exception propagates.
