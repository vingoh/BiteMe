# Memory Node Benchmark Design

**Date:** 2026-06-09  
**Status:** Approved

## Overview

A scored benchmark for evaluating the `memory_node` under different prompt configurations and LLM models. Cases are defined in external JSON files—**one file per scenario category** (`new_key.json`, `reuse.json`, etc.), each containing a `cases[]` array—for easy extension; pytest loads and flattens them, then runs real LLM calls. Each case is worth 10 points via one or more scoring rubric items; the final report shows per-case scores, a total out of 100, and a separate list of parse-failure cases.

## Goals

- Compare memory-node behavior across prompt variants and models
- Score key management (create / reuse / no-merge / multi-key), alias quality, score accuracy, and strength/weakness null handling
- Keep case content out of Python so new cases can be added without editing test logic
- Integrate with pytest (`@pytest.mark.integration`); CI runs default config only; local runs use `@pytest.mark.parametrize` for A/B comparison

## Non-Goals

- LLM-as-judge for subjective quality (all checks are programmatic)
- Automatic retry on LLM/parse failure
- Replacing `tests/test_memory_eval.py` (kept as quick observation tests during development)

## Architecture

```
tests/benchmarks/
  memory/
    cases/                    # one JSON file per scenario category
      new_key.json            # 2 cases
      reuse.json              # 2 cases
      no_merge.json           # 2 cases
      multi_key.json          # 2 cases
      scoring.json            # 2 cases
    schema.json               # optional JSON Schema for validation
  memory_scorer.py            # scoring check functions + runner helpers
  memory_loader.py            # load & validate JSON cases
tests/test_memory_benchmark.py  # pytest entry, parametrize configs
```

### Data Flow

```
JSON category files (cases/*.json)
    → memory_loader.load_cases()   # flatten cases[] across files
    → pytest (parametrize: config × case)
    → patch prompt/model, call memory_node (real LLM)
    → capture result: {written_memory, raw_updates, parse_ok}
    → memory_scorer.score(case, result)
    → per-case report + aggregate total + parse_failures list
```

## Case JSON Schema

Cases are grouped **one JSON file per scenario category**. Each file contains a `category` field and a `cases` array.

**File:** `tests/benchmarks/memory/cases/reuse.json`

```json
{
  "category": "reuse",
  "cases": [
    {
      "id": "reuse_context_window",
      "description": "相同概念应复用 llm_context_window，不新建 key",
      "initial_memory": {
        "entries": {
          "llm_context_window": {
            "aliases": ["上下文窗口", "context window", "token 限制"],
            "recent_scores": [7],
            "avg_score": 7.0,
            "last_update": "2026-01-01",
            "comments": { "strength": [], "weakness": [] }
          }
        }
      },
      "question": "大语言模型的上下文窗口（context window）是什么？...",
      "user_answer": "上下文窗口是 LLM 每次推理时能处理的最大 token 数量。...",
      "llm_reference": "上下文窗口（context window）指 LLM 在单次前向传播中能接收的最大 token 数。...",
      "scoring_points": [
        { "name": "reuse_correct_key", "points": 4, "check": "key_reused", "params": { "key": "llm_context_window" } },
        { "name": "no_new_keys",       "points": 3, "check": "no_new_keys" },
        { "name": "score_appended",    "points": 3, "check": "key_score_appended", "params": { "key": "llm_context_window" } }
      ]
    }
  ]
}
```

### Category File Fields (top level)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `category` | string | yes | One of: `new_key`, `reuse`, `no_merge`, `multi_key`, `scoring`; must match filename stem |
| `cases` | array | yes | List of case objects (see below) |

### Case Object Fields (inside `cases[]`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique case identifier **across all category files** |
| `description` | string | no | Human-readable note for reports |
| `initial_memory` | object | yes | Full `MemoryFile` structure passed to `save_memory` before the run |
| `question` | string | yes | Questioner message content |
| `user_answer` | string | yes | Human answer content |
| `llm_reference` | string | yes | Reference answer for the LLM scorer |
| `scoring_points` | array | yes | Rubric items; **must sum to 10** |

### Scoring Point Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Label shown in report |
| `points` | int | yes | Points awarded when check passes |
| `check` | string | yes | Registered check function name (see below) |
| `params` | object | no | Check-specific arguments |

## Registered Check Functions

Implemented in `memory_scorer.py`; referenced by `check` name in JSON.

### Key Management

| Check | Params | Pass Condition |
|-------|--------|----------------|
| `key_created` | `min_count` (default 1) | At least N new keys vs `initial_memory` |
| `key_reused` | `key` | Specified key received a new score entry |
| `key_score_appended` | `key` | Specified key's `recent_scores` length increased vs `initial_memory` |
| `key_not_updated` | `key` | Specified key's `recent_scores`, `last_update`, `comments` unchanged |
| `no_new_keys` | — | No keys added beyond `initial_memory` |
| `update_count_range` | `min`, `max` | Number of LLM updates in range |
| `key_semantic_match` | `keywords` | At least one new/reused key contains any keyword (case-insensitive) |
| `keys_semantically_distinct` | `min_count` (default 2) | At least N unique keys appear in LLM updates |
| `keyword_groups_covered` | `groups` (array of keyword arrays) | Each keyword group matches at least one distinct key; all matched keys must differ |

### Format & Quality

| Check | Params | Pass Condition |
|-------|--------|----------------|
| `snake_case_valid` | — | All keys in updates match `^[a-z][a-z0-9_]*$` |
| `no_broad_keys` | `blacklist` (optional) | No key in default blacklist: `basic_concept`, `mechanism_understanding`, `design_thinking` |
| `alias_nonempty` | `key` (optional) | New keys have ≥1 alias; if `key` given, check that key only |
| `alias_keywords` | `keywords`, `key` (optional) | At least one alias contains any keyword |

### Score & Comments

| Check | Params | Pass Condition |
|-------|--------|----------------|
| `score_near_standard` | `standard_score`, `threshold` (default 2), `key` (optional) | \|actual − standard\| ≤ threshold for the relevant update |
| `strength_is_null` | `key` (optional) | `strength` is null in LLM update |
| `strength_not_null` | `key` (optional) | `strength` is non-null |
| `weakness_is_null` | `key` (optional) | `weakness` is null in LLM update |
| `weakness_not_null` | `key` (optional) | `weakness` is non-null |
| `comment_not_generic` | `field` (`strength`/`weakness`), `key` (optional) | Non-null comment does not contain generic phrases: `回答较完整`, `表达清晰`, `理解不够深入`, `还需加强` |
| `comment_specific` | `field` (`strength`/`weakness`), `key` (optional) | Field is non-null **and** passes `comment_not_generic` (shorthand for scoring cases) |

### Parse / Execution

When `parse_ok` is false (LLM call or JSON parse failed), **all scoring points for that case score 0**. The case `id` is appended to `parse_failures` in the aggregate report. No retry.

## Case Catalog (10 cases, 100 points total)

Topics span **agent orchestration, tokenization, embeddings, prompting, alignment, inference optimization, and decoding** — not concentrated on RAG / context window alone.

| Case ID | Topic area |
|---------|------------|
| `new_key_langgraph_checkpointer` | Agent orchestration / HITL |
| `new_key_bpe_tokenization` | LLM tokenization (BPE) |
| `reuse_text_embedding` | Text embeddings |
| `reuse_precise_match_cot` | Chain-of-Thought prompting |
| `no_merge_rlhf_vs_sft` | RLHF vs supervised fine-tuning |
| `no_merge_kv_cache_vs_quantization` | KV cache vs model quantization |
| `multi_key_react_agent` | ReAct agent loop |
| `multi_key_llm_inference` | LLM inference optimization |
| `score_good_decoding` | Decoding strategies (temperature / top-p) |
| `score_poor_decoding` | Same topic, poor answer |

### Category 1: `new_key` — Create from empty memory

**`new_key_langgraph_checkpointer`** — Agent / HITL

| Scoring Point | Pts | Check |
|---------------|-----|-------|
| Creates ≥1 new key | 3 | `key_created` |
| All keys snake_case | 2 | `snake_case_valid` |
| New key has ≥1 alias | 2 | `alias_nonempty` |
| Key semantically relevant | 3 | `key_semantic_match` keywords: `checkpointer`, `langgraph`, `checkpoint` |

**`new_key_bpe_tokenization`** — LLM fundamentals

| Scoring Point | Pts | Check |
|---------------|-----|-------|
| Creates new key | 3 | `key_created` |
| No broad/generic keys | 3 | `no_broad_keys` |
| Alias includes relevant term | 2 | `alias_keywords` keywords: `token`, `BPE`, `分词`, `tokeniz` |
| Exactly 1 update | 2 | `update_count_range` min=1, max=1 |

### Category 2: `reuse` — Update existing key

**`reuse_text_embedding`**

Pre-loaded: `text_embedding`. Question about embedding models / semantic similarity.

| Scoring Point | Pts | Check |
|---------------|-----|-------|
| Reuses `text_embedding` | 4 | `key_reused` key=`text_embedding` |
| No new keys | 3 | `no_new_keys` |
| Score appended to entry | 3 | `key_score_appended` key=`text_embedding` |

**`reuse_precise_match_cot`**

Pre-loaded: `chain_of_thought`, `few_shot_learning`, `system_prompt_design`. Question about CoT prompting.

| Scoring Point | Pts | Check |
|---------------|-----|-------|
| Reuses `chain_of_thought` | 4 | `key_reused` key=`chain_of_thought` |
| `few_shot_learning` not updated | 2 | `key_not_updated` key=`few_shot_learning` |
| `system_prompt_design` not updated | 2 | `key_not_updated` key=`system_prompt_design` |
| Score in valid range | 2 | `score_near_standard` standard_score=6, threshold=4 |

### Category 3: `no_merge` — Similar topic, separate key

**`no_merge_rlhf_vs_sft`**

Pre-loaded: `supervised_finetuning`. Question about RLHF alignment pipeline.

| Scoring Point | Pts | Check |
|---------------|-----|-------|
| `supervised_finetuning` not updated | 4 | `key_not_updated` key=`supervised_finetuning` |
| New RLHF-related key created | 4 | `key_semantic_match` keywords: `rlhf`, `reward`, `偏好`, `alignment` |
| New key snake_case valid | 2 | `snake_case_valid` |

**`no_merge_kv_cache_vs_quantization`**

Pre-loaded: `model_quantization`. Question about KV cache in LLM inference.

| Scoring Point | Pts | Check |
|---------------|-----|-------|
| `model_quantization` not updated | 4 | `key_not_updated` key=`model_quantization` |
| New KV-cache key created | 4 | `key_semantic_match` keywords: `kv`, `cache`, `缓存` |
| No broad keys | 2 | `no_broad_keys` |

### Category 4: `multi_key` — Broad question, multiple knowledge points

**`multi_key_react_agent`**

Question covers ReAct loop + tool calling + observation feedback.

| Scoring Point | Pts | Check |
|---------------|-----|-------|
| 2–3 updates | 4 | `update_count_range` min=2, max=3 |
| ≥2 distinct keys | 3 | `keys_semantically_distinct` min_count=2 |
| No broad keys | 3 | `no_broad_keys` |

**`multi_key_llm_inference`**

Question covers KV cache + speculative decoding + continuous batching.

| Scoring Point | Pts | Check |
|---------------|-----|-------|
| ≥2 updates | 3 | `update_count_range` min=2, max=3 |
| Keys cover distinct sub-concepts | 4 | `keyword_groups_covered` groups: `[["kv","cache","缓存"], ["speculative","投机"], ["batch","批"]]` |
| Each new key has alias | 3 | `alias_nonempty` |

### Category 5: `scoring` — Score accuracy & comment quality

**`score_good_decoding`**

Topic: temperature / top-p / top-k sampling. User gives thorough answer. Standard score: **8**, threshold: **2**.

| Scoring Point | Pts | Check |
|---------------|-----|-------|
| Score within ±2 of 8 | 4 | `score_near_standard` standard_score=8, threshold=2 |
| Strength is specific (non-null, non-generic) | 3 | `comment_specific` field=strength |
| Weakness is null | 3 | `weakness_is_null` |

**`score_poor_decoding`**

Same decoding topic, vague wrong answer. Standard score: **2**, threshold: **2**.

| Scoring Point | Pts | Check |
|---------------|-----|-------|
| Score within ±2 of 2 | 4 | `score_near_standard` standard_score=2, threshold=2 |
| Weakness is specific (non-null, non-generic) | 3 | `comment_specific` field=weakness |
| Strength is null | 3 | `strength_is_null` |

## Pytest Integration

### File: `tests/test_memory_benchmark.py`

```python
DEFAULT_CONFIG = {"prompt_variant": "default", "model": None}  # None → settings.openai_model

# CI: pytest tests/test_memory_benchmark.py -m integration
# Local compare:
# @pytest.mark.parametrize("config", [
#     {"prompt_variant": "default", "model": "glm-5"},
#     {"prompt_variant": "v2", "model": "glm-5"},
# ], ids=["default", "v2"])
```

- `pytestmark = pytest.mark.integration` — skipped by default
- Fixture loads all category files from `tests/benchmarks/memory/cases/*.json`, flattens each file's `cases[]` into a single list
- Validates: `category` matches filename stem; all `id` values unique across files; each case's `sum(scoring_points.points) == 10`
- Patches `get_prompts()` to inject `prompt_variant` memory prompt
- Patches `settings.openai_model` when `config["model"]` is set
- Calls `memory_node` via existing `_make_state` pattern from `test_memory_eval.py`

### Capturing LLM Output

Extend the run helper to return:

```python
@dataclass
class BenchmarkResult:
    written_memory: dict
    updates: list[MemoryUpdate] | None   # parsed LLM output
    parse_ok: bool
```

Implementation: patch or wrap `memory_node` to also return parsed updates, **or** add a thin `run_memory_update(...)` helper in `memory_scorer.py` that duplicates the LLM call path but returns structured data. Prefer extracting a `invoke_memory_update(state, memory_path) -> BenchmarkResult` function from `memory.py` to avoid duplication.

### Report Format

Printed at end of session (`-s` flag):

```
=== Memory Benchmark [default] ===
new_key_langgraph_checkpointer      8/10  reuse_correct_key✓ snake_case✓ alias✓ semantic✗
reuse_context_window               10/10  ...
no_merge_rag_vs_context             0/10  [PARSE FAILED]
...
--------------------------------------------------
Total: 72/100 (72.0%)
Parse failures (1): no_merge_rag_vs_context
```

Pytest assertion: `total_score >= threshold` is **not** enforced (benchmark is observational). Optional `--benchmark-min-score=60` via custom pytest option for CI gating in the future.

## Prompt Variant Support

Add to `biteme/graph/prompts.py`:

```python
MEMORY_PROMPT_VARIANTS: dict[str, str] = {
    "default": MEMORY_UPDATER,
    # "v2": MEMORY_UPDATER_V2,  # added when experimenting
}

def get_memory_prompt(variant: str = "default") -> str:
    return MEMORY_PROMPT_VARIANTS[variant]
```

`get_prompts()` continues to use `"default"` in production. Benchmark patches `get_prompts` or `get_memory_prompt` per config.

## Adding a New Case

1. Open the matching category file, e.g. `tests/benchmarks/memory/cases/reuse.json`
2. Append a new object to the `cases` array (or create a new `<category>.json` if adding a new scenario)
3. Ensure `id` is unique across all files and `scoring_points` sum to 10
4. Run `conda run -n agent pytest tests/test_memory_benchmark.py -k <id> -m integration -s`
5. No Python changes needed unless a new check type is required

## Error Handling

| Situation | Behavior |
|-----------|----------|
| LLM call fails | `parse_ok=False`, case scores 0, id in `parse_failures` |
| JSON parse fails | Same as above |
| Case JSON invalid (points ≠ 10, duplicate id, category ≠ filename) | pytest collection error — fails fast at load time |
| Unknown `check` name in JSON | pytest collection error |

## Relationship to Existing Tests

| File | Role |
|------|------|
| `tests/test_memory.py` | Unit tests for `load_memory`, `apply_updates`, passthrough |
| `tests/test_memory_eval.py` | Informal integration tests for manual observation (unchanged) |
| `tests/test_memory_benchmark.py` | Scored benchmark with JSON cases (new) |

## Future Extensions (out of scope)

- `--benchmark-min-score` CI gate
- HTML/JSON report export for tracking scores over time
- JSON Schema validation in CI
- Retry on parse failure
