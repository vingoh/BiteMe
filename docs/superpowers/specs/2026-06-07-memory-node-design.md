# Memory Node Design

**Date:** 2026-06-07  
**Status:** Approved

## Overview

Add a `memory_node` to the LangGraph that runs after each HITL answerer turn. It uses an LLM to identify 1–3 knowledge points from the question/answer pair, scores the user's mastery (0–10), and persists results to a global JSON memory file.

## Graph Structure

**Before:**
```
planner → questioner → answerer → _should_continue → questioner | END
```

**After:**
```
planner → questioner → answerer → memory → _should_continue → questioner | END
```

`memory_node` is always in the graph. It checks `hitl_flags` internally and returns `{}` immediately when `"answerer"` is not in `hitl_flags` (non-HITL pass-through).

## Trigger Condition

Only active when `"answerer" in state["hitl_flags"]` (i.e., the user is the one answering). In LLM-only mode the node is a no-op.

## Inputs (from state, no new state fields needed)

| Data | Source |
|------|--------|
| Current question | `state["messages"][-2]["content"]` |
| User's answer | `state["messages"][-1]["content"]` |
| LLM reference answer | `state["llm_reference_answer"]` |
| Existing memory | loaded from `~/.biteme/memory.json` |

`memory_node` returns `{}` — it writes to disk, not to state.

## Memory File

**Path:** `~/.biteme/memory.json` (global, shared across all source paths)

**Schema:**
```json
{
  "entries": {
    "<snake_case_key>": {
      "aliases": ["语义标签1", "语义标签2"],
      "recent_scores": [6, 8, 7],
      "avg_score": 7.0,
      "last_update": "2026-06-07T16:10:00",
      "comments": {
        "strength": [
          "能描述 yield 基本行为"
        ],
        "weakness": [
          "未提及 StopIteration 和生成器协议"
        ]
      }
    }
  }
}
```

**Field rules:**
- `key`: snake_case identifier; LLM decides; prefer reusing existing keys
- `aliases`: semantic labels; LLM provides initial set at creation; new aliases may be appended in subsequent runs; duplicates are removed in code (order-preserving)
- `recent_scores`: unbounded list; each run appends one score per matched key
- `avg_score`: computed by Python as `mean(recent_scores)` on every write
- `last_update`: date string in `YYYY-MM-DD` format; updated on every write
- `comments.strength`: list of strength strings, one appended per run
- `comments.weakness`: list of weakness strings, one appended per run

## LLM Interaction

### Structured Output Schema (Pydantic)

```python
class MemoryUpdate(BaseModel):
    key: str              # snake_case; existing key or new one
    aliases: list[str]    # if key is new: initial aliases; if key exists: new aliases to add (may be empty)
    score: int            # 0–10
    strength: str | None  # specific strength in this answer; null if none identifiable
    weakness: str | None  # specific error/omission/unclear point; null if none identifiable

class MemoryUpdates(BaseModel):
    updates: list[MemoryUpdate]  # 1–3 items
```

Whether a key is new is determined in code by checking `key in data["entries"]`, not by LLM output.

### Prompt

```
你是一个知识点追踪助手。根据提供的一问一答，识别其中涉及的 1–3 个知识点，
并评估用户对每个知识点的掌握程度。

## 已有知识点
{existing_keys}

每个已有知识点包含：
- key：稳定标识符
- aliases：该知识点的别名或同义说法

## 本轮问答
问题：{question}
用户回答：{user_answer}
LLM 参考答案：{llm_reference}

## 任务
1. 根据"问题"和"LLM 参考答案"，识别本轮真正考察的 1–3 个核心知识点。
2. 对每个知识点，先尝试匹配"已有知识点"：
   - 如果当前知识点与某个已有 key / aliases 表示的是同一个可学习概念，必须复用已有 key。
   - 如果只是话题相关、上下游相关、或者名称相似但考察重点不同，不要复用。
   - 如果没有合适的已有 key，创建新的 snake_case key。
3. 根据"用户回答"相对于"LLM 参考答案"的准确性、完整性和深度，对每个知识点打 0–10 分。
4. 给出一句 strength 和一句 weakness，必须具体对应用户本轮回答，不要泛泛而谈。

## 约束
- items 数量为 1–3。
- key 必须是英文小写 snake_case。
- 如果复用已有 key，key 必须与已有知识点中的 key 完全一致。
- aliases 只放本轮新出现、且有助于后续匹配的别名；没有则为空数组。
- score 必须是 0–10 的整数。
- strength：只有当用户回答中存在具体、可定位的优点时才输出；否则为 null。禁止泛泛评价，例如"回答较完整""表达清晰"。
- weakness：只有当用户回答中存在具体错误、遗漏或表达不清时才输出；否则为 null。禁止泛泛评价，例如"理解不够深入""还需加强"。
- 不要输出过宽泛的 key，例如 mechanism_understanding、design_thinking、basic_concept。
```

## Write Logic (`save_memory`)

For each `MemoryUpdate`:

```
if update.key not in data["entries"]:
    create entry with aliases, recent_scores=[], comments={strength:[], weakness:[]}
else:
    merge aliases (set-based dedup, order-preserving):
        existing_set = set(entry["aliases"])
        for alias in update.aliases:
            if alias not in existing_set:
                entry["aliases"].append(alias)
                existing_set.add(alias)

append update.score to entry["recent_scores"]
entry["avg_score"] = mean(entry["recent_scores"])
entry["last_update"] = date.today().isoformat()  # YYYY-MM-DD
if update.strength is not None:
    entry["comments"]["strength"].append(update.strength)
if update.weakness is not None:
    entry["comments"]["weakness"].append(update.weakness)
```

**Atomic write:** write to a temp file in the same directory, then `os.replace(tmp, target)` to prevent corruption on interrupted writes.

## Code Structure

**New file:** `biteme/graph/memory.py`

Contains:
- `MemoryEntry` (TypedDict) — single entry structure
- `MemoryUpdate`, `MemoryUpdates` (Pydantic BaseModel) — LLM output schema
- `load_memory(path) -> dict` — read JSON; return `{"entries": {}}` if file absent
- `save_memory(data, path)` — atomic write
- `memory_node(state) -> dict` — LangGraph node function

**Modified files:**
- `biteme/graph/prompts.py` — add `MEMORY_UPDATER` prompt constant and expose via `get_prompts()`
- `biteme/graph/graph.py` — add `memory_node` import, register node, add edge `answerer → memory`
- `biteme/graph/nodes.py` — no changes needed
- `biteme/config.py` — no changes needed (`memory.json` path derived from existing `biteme_home`)

## Testing

**New file:** `tests/test_memory.py`

Unit tests:
- `load_memory` returns `{"entries": {}}` when file does not exist
- `save_memory` atomic write: written file reads back identically
- `memory_node` with non-HITL state returns `{}` and does not touch the file
- alias deduplication: adding existing alias does not create duplicate
- `avg_score` computed correctly after multiple appends
