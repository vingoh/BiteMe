# Memory Node Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scored pytest benchmark that loads memory cases from per-category JSON files, runs real LLM calls against `memory_node`, and reports per-case scores (10 pts each, 100 total) plus parse-failure tracking.

**Architecture:** Extract `invoke_memory_update()` from `memory.py` so the benchmark can capture parsed LLM output without duplicating the call path. `memory_loader.py` flattens `cases/*.json`; `memory_scorer.py` registers programmatic check functions referenced by name in JSON. `test_memory_benchmark.py` parametrize configs for local A/B comparison; default config only in CI.

**Tech Stack:** Python 3.10+, pytest, Pydantic v2, LangChain `ChatOpenAI`, existing `biteme.graph.memory` module

**Spec:** `docs/superpowers/specs/2026-06-09-memory-benchmark-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `biteme/graph/memory.py` | Extract `invoke_memory_update`, refactor `memory_node` to use it |
| Modify | `biteme/graph/prompts.py` | Add `MEMORY_PROMPT_VARIANTS`, `get_memory_prompt()` |
| Create | `tests/benchmarks/memory_loader.py` | Load & validate category JSON files |
| Create | `tests/benchmarks/memory_scorer.py` | Check registry, `score_case()`, `BenchmarkResult` |
| Create | `tests/benchmarks/memory/cases/new_key.json` | 2 cases |
| Create | `tests/benchmarks/memory/cases/reuse.json` | 2 cases |
| Create | `tests/benchmarks/memory/cases/no_merge.json` | 2 cases |
| Create | `tests/benchmarks/memory/cases/multi_key.json` | 2 cases |
| Create | `tests/benchmarks/memory/cases/scoring.json` | 2 cases |
| Create | `tests/test_memory_benchmark_loader.py` | Unit tests for loader + scorer (no LLM) |
| Create | `tests/test_memory_benchmark.py` | Integration benchmark runner |
| Modify | `tests/test_memory.py` | Add test for `invoke_memory_update` passthrough (optional, covered in Task 2) |

---

## Task 1: Case Loader

**Files:**
- Create: `tests/benchmarks/memory_loader.py`
- Create: `tests/benchmarks/memory/cases/.gitkeep` (temporary; replaced by JSON in Task 5)
- Create: `tests/test_memory_benchmark_loader.py`

- [ ] **Step 1: Write failing loader tests**

Create `tests/test_memory_benchmark_loader.py`:

```python
import json
import pytest
from pathlib import Path

from tests.benchmarks.memory_loader import load_cases, MemoryBenchmarkCase

CASES_DIR = Path(__file__).parent / "benchmarks" / "memory" / "cases"


def _write_category(tmp_path: Path, name: str, cases: list) -> None:
    (tmp_path / f"{name}.json").write_text(
        json.dumps({"category": name, "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )


def _minimal_case(case_id: str, points: int = 10) -> dict:
    return {
        "id": case_id,
        "initial_memory": {"entries": {}},
        "question": "q",
        "user_answer": "a",
        "llm_reference": "r",
        "scoring_points": [
            {"name": "only", "points": points, "check": "key_created"}
        ],
    }


def test_load_cases_flattens_multiple_files(tmp_path):
    _write_category(tmp_path, "new_key", [_minimal_case("case_a")])
    _write_category(tmp_path, "reuse", [_minimal_case("case_b")])
    cases = load_cases(tmp_path)
    assert {c.id for c in cases} == {"case_a", "case_b"}


def test_load_cases_rejects_points_not_summing_to_10(tmp_path):
    _write_category(tmp_path, "new_key", [_minimal_case("bad", points=7)])
    with pytest.raises(ValueError, match="must sum to 10"):
        load_cases(tmp_path)


def test_load_cases_rejects_duplicate_ids(tmp_path):
    _write_category(tmp_path, "new_key", [_minimal_case("dup")])
    _write_category(tmp_path, "reuse", [_minimal_case("dup")])
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(tmp_path)


def test_load_cases_rejects_category_filename_mismatch(tmp_path):
    (tmp_path / "reuse.json").write_text(
        json.dumps({"category": "new_key", "cases": [_minimal_case("x")]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="category"):
        load_cases(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n agent pytest tests/test_memory_benchmark_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError: tests.benchmarks.memory_loader`

- [ ] **Step 3: Implement `memory_loader.py`**

Create `tests/benchmarks/__init__.py` (empty) and `tests/benchmarks/memory_loader.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScoringPoint:
    name: str
    points: int
    check: str
    params: dict = field(default_factory=dict)


@dataclass
class MemoryBenchmarkCase:
    id: str
    category: str
    initial_memory: dict
    question: str
    user_answer: str
    llm_reference: str
    scoring_points: list[ScoringPoint]
    description: str = ""


def load_cases(cases_dir: Path) -> list[MemoryBenchmarkCase]:
    cases: list[MemoryBenchmarkCase] = []
    seen_ids: set[str] = set()

    for path in sorted(cases_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        category = data["category"]
        if category != path.stem:
            raise ValueError(
                f"{path.name}: category '{category}' does not match filename stem '{path.stem}'"
            )
        for raw in data["cases"]:
            case_id = raw["id"]
            if case_id in seen_ids:
                raise ValueError(f"duplicate case id: {case_id}")
            seen_ids.add(case_id)

            scoring_points = [
                ScoringPoint(
                    name=sp["name"],
                    points=sp["points"],
                    check=sp["check"],
                    params=sp.get("params", {}),
                )
                for sp in raw["scoring_points"]
            ]
            total = sum(sp.points for sp in scoring_points)
            if total != 10:
                raise ValueError(f"case {case_id}: scoring_points must sum to 10, got {total}")

            cases.append(
                MemoryBenchmarkCase(
                    id=case_id,
                    category=category,
                    initial_memory=raw["initial_memory"],
                    question=raw["question"],
                    user_answer=raw["user_answer"],
                    llm_reference=raw["llm_reference"],
                    scoring_points=scoring_points,
                    description=raw.get("description", ""),
                )
            )
    return cases
```

- [ ] **Step 4: Run loader tests**

```bash
conda run -n agent pytest tests/test_memory_benchmark_loader.py -v
```

Expected: PASS (4 tests)

---

## Task 2: Extract `invoke_memory_update` from `memory.py`

**Files:**
- Modify: `biteme/graph/memory.py`
- Modify: `tests/test_memory_benchmark_loader.py` (add import smoke test)

- [ ] **Step 1: Add `MemoryInvokeResult` dataclass and `invoke_memory_update`**

Add near top of `biteme/graph/memory.py` (after imports):

```python
from dataclasses import dataclass
```

Add before `memory_node`:

```python
@dataclass
class MemoryInvokeResult:
    updates: list[MemoryUpdate] | None
    parse_ok: bool


def invoke_memory_update(
    *,
    existing_keys: list[dict],
    question: str,
    user_answer: str,
    llm_reference: str,
    mode: str,
    model: str | None = None,
) -> MemoryInvokeResult:
    """Call the memory LLM and return parsed updates. Does not touch disk."""
    prompts = get_prompts(mode)
    prompt_text = prompts["memory"].format(
        existing_keys=existing_keys,
        question=question,
        user_answer=user_answer,
        llm_reference=llm_reference,
    )
    llm_model = model or settings.openai_model
    llm = ChatOpenAI(model=llm_model, temperature=0.0)
    structured_llm = llm.with_structured_output(MemoryUpdates)

    result: MemoryUpdates | None = None
    try:
        result = structured_llm.invoke([HumanMessage(content=prompt_text)])
        return MemoryInvokeResult(updates=result.updates, parse_ok=True)
    except Exception as exc:
        raw_content: str | None = None
        if isinstance(exc, ValidationError):
            for err in exc.errors():
                if err.get("type") == "json_invalid" and isinstance(err.get("input"), str):
                    raw_content = err["input"]
                    break
        if raw_content is not None:
            try:
                result = _parse_memory_updates(raw_content)
                return MemoryInvokeResult(updates=result.updates, parse_ok=True)
            except Exception:
                pass
        try:
            raw_resp = llm.invoke([HumanMessage(content=prompt_text)])
            result = _parse_memory_updates(raw_resp.content)
            return MemoryInvokeResult(updates=result.updates, parse_ok=True)
        except Exception:
            logger.exception("invoke_memory_update LLM call failed")
            return MemoryInvokeResult(updates=None, parse_ok=False)
```

- [ ] **Step 2: Refactor `memory_node` to use `invoke_memory_update`**

Replace the LLM call block inside `memory_node` with:

```python
    invoke_result = invoke_memory_update(
        existing_keys=existing_keys,
        question=question,
        user_answer=user_answer,
        llm_reference=llm_reference,
        mode=state["mode"],
    )
    if not invoke_result.parse_ok or invoke_result.updates is None:
        return {}

    apply_updates(data, invoke_result.updates)
    save_memory(data, memory_path)

    console = Console()
    summary = "\n".join(
        f"  {u.key} score={u.score}  strength: {u.strength}  weakness: {u.weakness}"
        for u in invoke_result.updates
    )
    console.print(Panel(summary, title="[magenta]Memory Updated[/magenta]"))
    return {}
```

Remove the old try/except LLM block and the duplicate `result` variable usage.

- [ ] **Step 3: Run existing unit tests**

```bash
conda run -n agent pytest tests/test_memory.py -v
```

Expected: PASS (all existing tests)

---

## Task 3: Scorer — Check Registry and `score_case`

**Files:**
- Create: `tests/benchmarks/memory_scorer.py`
- Modify: `tests/test_memory_benchmark_loader.py` (add scorer unit tests)

- [ ] **Step 1: Write failing scorer tests**

Append to `tests/test_memory_benchmark_loader.py`:

```python
from biteme.graph.memory import MemoryUpdate
from tests.benchmarks.memory_scorer import BenchmarkResult, score_case
from tests.benchmarks.memory_loader import MemoryBenchmarkCase, ScoringPoint


def _case(**kwargs) -> MemoryBenchmarkCase:
    defaults = dict(
        id="test",
        category="new_key",
        initial_memory={"entries": {}},
        question="q",
        user_answer="a",
        llm_reference="r",
        scoring_points=[ScoringPoint("k", 10, "key_created")],
    )
    defaults.update(kwargs)
    return MemoryBenchmarkCase(**defaults)


def test_score_case_parse_failure_scores_zero():
    case = _case()
    result = BenchmarkResult(
        written_memory={"entries": {}},
        updates=None,
        parse_ok=False,
    )
    scored = score_case(case, result)
    assert scored.earned == 0
    assert scored.max_points == 10


def test_score_case_key_reused():
    case = _case(
        initial_memory={
            "entries": {
                "foo": {
                    "aliases": [],
                    "recent_scores": [5],
                    "avg_score": 5.0,
                    "last_update": "2026-01-01",
                    "comments": {"strength": [], "weakness": []},
                }
            }
        },
        scoring_points=[
            ScoringPoint("reuse", 10, "key_reused", {"key": "foo"}),
        ],
    )
    result = BenchmarkResult(
        written_memory={
            "entries": {
                "foo": {
                    "aliases": [],
                    "recent_scores": [5, 7],
                    "avg_score": 6.0,
                    "last_update": "2026-06-09",
                    "comments": {"strength": [], "weakness": []},
                }
            }
        },
        updates=[MemoryUpdate(key="foo", aliases=[], score=7, strength=None, weakness=None)],
        parse_ok=True,
    )
    scored = score_case(case, result)
    assert scored.earned == 10


def test_score_near_standard_within_threshold():
    case = _case(
        scoring_points=[
            ScoringPoint("score", 10, "score_near_standard", {
                "standard_score": 8, "threshold": 2
            }),
        ],
    )
    result = BenchmarkResult(
        written_memory={"entries": {}},
        updates=[MemoryUpdate(key="x", aliases=[], score=7, strength=None, weakness=None)],
        parse_ok=True,
    )
    scored = score_case(case, result)
    assert scored.earned == 10
```

- [ ] **Step 2: Run scorer tests to verify they fail**

```bash
conda run -n agent pytest tests/test_memory_benchmark_loader.py::test_score_case_parse_failure_scores_zero -v
```

Expected: FAIL — `ModuleNotFoundError: tests.benchmarks.memory_scorer`

- [ ] **Step 3: Implement `memory_scorer.py`**

```python
from __future__ import annotations

import re
from dataclasses import dataclass, field

from biteme.graph.memory import MemoryUpdate
from tests.benchmarks.memory_loader import MemoryBenchmarkCase

SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
DEFAULT_BROAD_KEYS = {"basic_concept", "mechanism_understanding", "design_thinking"}
GENERIC_PHRASES = ("回答较完整", "表达清晰", "理解不够深入", "还需加强")


@dataclass
class BenchmarkResult:
    written_memory: dict
    updates: list[MemoryUpdate] | None
    parse_ok: bool


@dataclass
class ScoredPoint:
    name: str
    points: int
    earned: int
    passed: bool


@dataclass
class CaseScore:
    case_id: str
    earned: int
    max_points: int
    points: list[ScoredPoint] = field(default_factory=list)
    parse_failed: bool = False


def _initial_entries(case: MemoryBenchmarkCase) -> dict:
    return case.initial_memory.get("entries", {})


def _new_keys(case: MemoryBenchmarkCase, result: BenchmarkResult) -> set[str]:
    before = set(_initial_entries(case))
    after = set(result.written_memory.get("entries", {}))
    return after - before


def _get_update(updates: list[MemoryUpdate], key: str | None) -> MemoryUpdate | None:
    if not updates:
        return None
    if key:
        for u in updates:
            if u.key == key:
                return u
        return None
    return updates[0]


def _check_key_created(case, result, params) -> bool:
    min_count = params.get("min_count", 1)
    return len(_new_keys(case, result)) >= min_count


def _check_key_reused(case, result, params) -> bool:
    key = params["key"]
    entry = result.written_memory.get("entries", {}).get(key)
    if not entry:
        return False
    initial = _initial_entries(case).get(key, {})
    return len(entry.get("recent_scores", [])) > len(initial.get("recent_scores", []))


def _check_key_score_appended(case, result, params) -> bool:
    return _check_key_reused(case, result, params)


def _check_key_not_updated(case, result, params) -> bool:
    key = params["key"]
    before = _initial_entries(case).get(key)
    after = result.written_memory.get("entries", {}).get(key)
    if before is None or after is None:
        return before == after
    return (
        after["recent_scores"] == before["recent_scores"]
        and after["last_update"] == before["last_update"]
        and after["comments"] == before["comments"]
    )


def _check_no_new_keys(case, result, _params) -> bool:
    return len(_new_keys(case, result)) == 0


def _check_update_count_range(_case, result, params) -> bool:
    if not result.updates:
        return False
    n = len(result.updates)
    return params.get("min", 0) <= n <= params.get("max", 999)


def _check_key_semantic_match(case, result, params) -> bool:
    keywords = [k.lower() for k in params["keywords"]]
    keys_to_check: set[str] = set()
    if result.updates:
        keys_to_check.update(u.key for u in result.updates)
    keys_to_check.update(_new_keys(case, result))
    for key in keys_to_check:
        kl = key.lower()
        if any(kw in kl for kw in keywords):
            return True
    if result.updates:
        for u in result.updates:
            for alias in u.aliases:
                al = alias.lower()
                if any(kw in al for kw in keywords):
                    return True
    return False


def _check_keys_semantically_distinct(_case, result, params) -> bool:
    if not result.updates:
        return False
    min_count = params.get("min_count", 2)
    return len({u.key for u in result.updates}) >= min_count


def _check_keyword_groups_covered(_case, result, params) -> bool:
    if not result.updates:
        return False
    matched_keys: list[str] = []
    for group in params["groups"]:
        group_l = [g.lower() for g in group]
        found_key = None
        for u in result.updates:
            text = u.key.lower() + " " + " ".join(a.lower() for a in u.aliases)
            if any(g in text for g in group_l):
                found_key = u.key
                break
        if found_key is None:
            return False
        matched_keys.append(found_key)
    return len(set(matched_keys)) == len(matched_keys)


def _check_snake_case_valid(_case, result, _params) -> bool:
    if not result.updates:
        return False
    return all(SNAKE_CASE_RE.match(u.key) for u in result.updates)


def _check_no_broad_keys(_case, result, params) -> bool:
    if not result.updates:
        return False
    blacklist = set(params.get("blacklist", [])) | DEFAULT_BROAD_KEYS
    return not any(u.key in blacklist for u in result.updates)


def _check_alias_nonempty(case, result, params) -> bool:
    if not result.updates:
        return False
    target_key = params.get("key")
    updates = result.updates
    if target_key:
        updates = [u for u in updates if u.key == target_key]
    new_keys = _new_keys(case, result)
    checks = [u for u in updates if u.key in new_keys or target_key]
    if not checks:
        return False
    return all(len(u.aliases) >= 1 for u in checks)


def _check_alias_keywords(case, result, params) -> bool:
    keywords = [k.lower() for k in params["keywords"]]
    target_key = params.get("key")
    updates = result.updates or []
    if target_key:
        updates = [u for u in updates if u.key == target_key]
    for u in updates:
        for alias in u.aliases:
            if any(kw in alias.lower() for kw in keywords):
                return True
    return False


def _check_score_near_standard(_case, result, params) -> bool:
    if not result.updates:
        return False
    standard = params["standard_score"]
    threshold = params.get("threshold", 2)
    key = params.get("key")
    u = _get_update(result.updates, key)
    if u is None:
        return False
    return abs(u.score - standard) <= threshold


def _check_strength_is_null(_case, result, params) -> bool:
    u = _get_update(result.updates or [], params.get("key"))
    return u is not None and u.strength is None


def _check_strength_not_null(_case, result, params) -> bool:
    u = _get_update(result.updates or [], params.get("key"))
    return u is not None and u.strength is not None


def _check_weakness_is_null(_case, result, params) -> bool:
    u = _get_update(result.updates or [], params.get("key"))
    return u is not None and u.weakness is None


def _check_weakness_not_null(_case, result, params) -> bool:
    u = _get_update(result.updates or [], params.get("key"))
    return u is not None and u.weakness is not None


def _check_comment_not_generic(_case, result, params) -> bool:
    field = params["field"]
    u = _get_update(result.updates or [], params.get("key"))
    if u is None:
        return False
    text = getattr(u, field)
    if text is None:
        return False
    return not any(phrase in text for phrase in GENERIC_PHRASES)


def _check_comment_specific(case, result, params) -> bool:
    field = params["field"]
    u = _get_update(result.updates or [], params.get("key"))
    if u is None or getattr(u, field) is None:
        return False
    return _check_comment_not_generic(case, result, params)


CHECKS: dict[str, callable] = {
    "key_created": _check_key_created,
    "key_reused": _check_key_reused,
    "key_score_appended": _check_key_score_appended,
    "key_not_updated": _check_key_not_updated,
    "no_new_keys": _check_no_new_keys,
    "update_count_range": _check_update_count_range,
    "key_semantic_match": _check_key_semantic_match,
    "keys_semantically_distinct": _check_keys_semantically_distinct,
    "keyword_groups_covered": _check_keyword_groups_covered,
    "snake_case_valid": _check_snake_case_valid,
    "no_broad_keys": _check_no_broad_keys,
    "alias_nonempty": _check_alias_nonempty,
    "alias_keywords": _check_alias_keywords,
    "score_near_standard": _check_score_near_standard,
    "strength_is_null": _check_strength_is_null,
    "strength_not_null": _check_strength_not_null,
    "weakness_is_null": _check_weakness_is_null,
    "weakness_not_null": _check_weakness_not_null,
    "comment_not_generic": _check_comment_not_generic,
    "comment_specific": _check_comment_specific,
}


def validate_check_names(case: MemoryBenchmarkCase) -> None:
    for sp in case.scoring_points:
        if sp.check not in CHECKS:
            raise ValueError(f"case {case.id}: unknown check '{sp.check}'")


def score_case(case: MemoryBenchmarkCase, result: BenchmarkResult) -> CaseScore:
    max_points = sum(sp.points for sp in case.scoring_points)
    if not result.parse_ok:
        return CaseScore(
            case_id=case.id,
            earned=0,
            max_points=max_points,
            points=[
                ScoredPoint(sp.name, sp.points, 0, False) for sp in case.scoring_points
            ],
            parse_failed=True,
        )
    scored_points: list[ScoredPoint] = []
    earned = 0
    for sp in case.scoring_points:
        fn = CHECKS[sp.check]
        passed = fn(case, result, sp.params)
        pt_earned = sp.points if passed else 0
        earned += pt_earned
        scored_points.append(ScoredPoint(sp.name, sp.points, pt_earned, passed))
    return CaseScore(case_id=case.id, earned=earned, max_points=max_points, points=scored_points)
```

- [ ] **Step 4: Run all loader/scorer unit tests**

```bash
conda run -n agent pytest tests/test_memory_benchmark_loader.py -v
```

Expected: PASS (7 tests)

---

## Task 4: Prompt Variant Support

**Files:**
- Modify: `biteme/graph/prompts.py`

- [ ] **Step 1: Add variant registry**

At bottom of `biteme/graph/prompts.py`, before `get_prompts`:

```python
MEMORY_PROMPT_VARIANTS: dict[str, str] = {
    "default": MEMORY_UPDATER,
}


def get_memory_prompt(variant: str = "default") -> str:
    if variant not in MEMORY_PROMPT_VARIANTS:
        raise KeyError(f"Unknown memory prompt variant: {variant}")
    return MEMORY_PROMPT_VARIANTS[variant]
```

Update `get_prompts` to use `get_memory_prompt("default")` for the `"memory"` key:

```python
        "memory": get_memory_prompt("default"),
```

(and same for interview mode return dict)

- [ ] **Step 2: Verify imports still work**

```bash
conda run -n agent python -c "from biteme.graph.prompts import get_memory_prompt; print(get_memory_prompt('default')[:40])"
```

Expected: prints first 40 chars of prompt, no error

---

## Task 5: JSON Case Files (5 category files, 10 cases)

**Files:**
- Create: `tests/benchmarks/memory/cases/new_key.json`
- Create: `tests/benchmarks/memory/cases/reuse.json`
- Create: `tests/benchmarks/memory/cases/no_merge.json`
- Create: `tests/benchmarks/memory/cases/multi_key.json`
- Create: `tests/benchmarks/memory/cases/scoring.json`

- [ ] **Step 1: Create `new_key.json`**

```json
{
  "category": "new_key",
  "cases": [
    {
      "id": "new_key_langgraph_checkpointer",
      "description": "空 memory 应新建 LangGraph checkpointer 相关 key",
      "initial_memory": { "entries": {} },
      "question": "LangGraph 中 checkpointer 的作用是什么？它是如何实现持久化的？",
      "user_answer": "checkpointer 用于保存图执行过程中的状态快照，支持从中断点恢复执行，也就是说可以在 HITL 场景下暂停等待用户输入。常见实现有 SqliteSaver（本地 SQLite）和 PostgresSaver（云端）。",
      "llm_reference": "LangGraph 的 checkpointer 在每个节点执行后将当前 state 序列化并持久化存储。当 graph 遇到 interrupt 时，执行挂起；外部输入到达后可从对应 checkpoint 恢复。这使 Human-in-the-Loop（HITL）场景成为可能，也支持长时运行的 agent 故障恢复。接口：BaseCheckpointSaver，内置实现包括 SqliteSaver、AsyncSqliteSaver。",
      "scoring_points": [
        { "name": "creates_key", "points": 3, "check": "key_created" },
        { "name": "snake_case", "points": 2, "check": "snake_case_valid" },
        { "name": "has_alias", "points": 2, "check": "alias_nonempty" },
        { "name": "semantic", "points": 3, "check": "key_semantic_match", "params": { "keywords": ["checkpointer", "langgraph", "checkpoint"] } }
      ]
    },
    {
      "id": "new_key_bpe_tokenization",
      "description": "空 memory 应新建 BPE/tokenization 相关 key",
      "initial_memory": { "entries": {} },
      "question": "大语言模型使用的 BPE（Byte Pair Encoding）分词算法是如何工作的？为什么 LLM 需要子词（subword）分词？",
      "user_answer": "BPE 从字符级词表出发，迭代合并最高频的相邻 token 对，逐步构建子词词表。子词分词能平衡词表大小与未登录词（OOV）问题，常见词保持完整，罕见词拆成子词片段。",
      "llm_reference": "BPE 通过统计语料中相邻符号对的频率，反复合并最高频对来扩展词表，直到达到预设词表大小。子词分词使模型能处理任意新词（通过子词组合），同时控制 embedding 矩阵规模。GPT 系列使用 BPE；SentencePiece 等变体支持多语言和无空格语言。",
      "scoring_points": [
        { "name": "creates_key", "points": 3, "check": "key_created" },
        { "name": "no_broad", "points": 3, "check": "no_broad_keys" },
        { "name": "alias_terms", "points": 2, "check": "alias_keywords", "params": { "keywords": ["token", "BPE", "分词", "tokeniz"] } },
        { "name": "single_update", "points": 2, "check": "update_count_range", "params": { "min": 1, "max": 1 } }
      ]
    }
  ]
}
```

- [ ] **Step 2: Create `reuse.json`**

```json
{
  "category": "reuse",
  "cases": [
    {
      "id": "reuse_text_embedding",
      "description": "相同概念应复用 text_embedding，不新建 key",
      "initial_memory": {
        "entries": {
          "text_embedding": {
            "aliases": ["文本嵌入", "text embedding", "向量表示"],
            "recent_scores": [7],
            "avg_score": 7.0,
            "last_update": "2026-01-01",
            "comments": { "strength": ["能说明 embedding 的基本用途"], "weakness": ["未提及不同 embedding 模型的差异"] }
          }
        }
      },
      "question": "文本 embedding 模型是如何将自然语言转换为向量的？它在语义搜索中起什么作用？",
      "user_answer": "Embedding 模型通过预训练将文本映射到高维向量空间，语义相近的文本向量距离更近。在语义搜索中，query 和文档都转成向量，用余弦相似度或点积检索最相关的文档。",
      "llm_reference": "文本 embedding 模型（如 text-embedding-ada-002、BGE、E5）通过对比学习或掩码语言模型预训练，将变长文本编码为固定维度向量。语义搜索中，query embedding 与文档 embedding 计算相似度，检索 top-k 结果，比关键词匹配更能捕获语义相关性。",
      "scoring_points": [
        { "name": "reuse_key", "points": 4, "check": "key_reused", "params": { "key": "text_embedding" } },
        { "name": "no_new_keys", "points": 3, "check": "no_new_keys" },
        { "name": "score_appended", "points": 3, "check": "key_score_appended", "params": { "key": "text_embedding" } }
      ]
    },
    {
      "id": "reuse_precise_match_cot",
      "description": "多 key 场景应精准命中 chain_of_thought",
      "initial_memory": {
        "entries": {
          "chain_of_thought": {
            "aliases": ["思维链", "chain of thought", "CoT"],
            "recent_scores": [6],
            "avg_score": 6.0,
            "last_update": "2026-01-01",
            "comments": { "strength": [], "weakness": [] }
          },
          "few_shot_learning": {
            "aliases": ["少样本学习", "few-shot", "in-context learning"],
            "recent_scores": [7],
            "avg_score": 7.0,
            "last_update": "2026-01-01",
            "comments": { "strength": [], "weakness": [] }
          },
          "system_prompt_design": {
            "aliases": ["系统提示词", "system prompt", "角色设定"],
            "recent_scores": [8],
            "avg_score": 8.0,
            "last_update": "2026-01-01",
            "comments": { "strength": [], "weakness": [] }
          }
        }
      },
      "question": "Chain-of-Thought（思维链）提示是如何提升 LLM 推理能力的？与直接提问相比有什么优势？",
      "user_answer": "CoT 引导模型在给出最终答案前先输出中间推理步骤，把复杂问题分解为可验证的子步骤。相比直接提问，CoT 能显著提升数学推理、逻辑推理等需要多步思考的问题的准确率。",
      "llm_reference": "Chain-of-Thought 通过在 prompt 中加入「让我们一步步思考」或提供推理范例，诱导模型生成中间推理链再得出答案。这激活了模型内部的推理能力，在 GSM8K 等基准上大幅提升准确率。Zero-shot CoT 只需添加触发短语；Few-shot CoT 提供推理范例效果更好。",
      "scoring_points": [
        { "name": "reuse_cot", "points": 4, "check": "key_reused", "params": { "key": "chain_of_thought" } },
        { "name": "fewshot_unchanged", "points": 2, "check": "key_not_updated", "params": { "key": "few_shot_learning" } },
        { "name": "system_unchanged", "points": 2, "check": "key_not_updated", "params": { "key": "system_prompt_design" } },
        { "name": "score_sane", "points": 2, "check": "score_near_standard", "params": { "standard_score": 6, "threshold": 4 } }
      ]
    }
  ]
}
```

- [ ] **Step 3: Create `no_merge.json`**

```json
{
  "category": "no_merge",
  "cases": [
    {
      "id": "no_merge_rlhf_vs_sft",
      "description": "RLHF 与 SFT 相关但考察重点不同，不应合并",
      "initial_memory": {
        "entries": {
          "supervised_finetuning": {
            "aliases": ["监督微调", "SFT", "supervised fine-tuning"],
            "recent_scores": [6],
            "avg_score": 6.0,
            "last_update": "2026-01-01",
            "comments": { "strength": [], "weakness": [] }
          }
        }
      },
      "question": "RLHF（基于人类反馈的强化学习）是如何对齐大语言模型的？它与 SFT 有什么不同？",
      "user_answer": "RLHF 先训练奖励模型学习人类偏好，再用 PPO 等强化学习算法优化 LLM 策略，使输出更符合人类期望。SFT 只是用标注数据做监督学习；RLHF 在此基础上进一步优化难以用监督损失表达的偏好（如有用性、安全性）。",
      "llm_reference": "RLHF 三阶段：1) SFT 在高质量示范数据上微调；2) 训练 reward model 预测人类偏好排序；3) 用 PPO 以 reward model 为信号优化策略。相比纯 SFT，RLHF 能优化开放性质量维度（helpfulness、harmlessness）。InstructGPT、ChatGPT 均采用此流程。",
      "scoring_points": [
        { "name": "sft_not_updated", "points": 4, "check": "key_not_updated", "params": { "key": "supervised_finetuning" } },
        { "name": "rlhf_key", "points": 4, "check": "key_semantic_match", "params": { "keywords": ["rlhf", "reward", "偏好", "alignment"] } },
        { "name": "snake_case", "points": 2, "check": "snake_case_valid" }
      ]
    },
    {
      "id": "no_merge_kv_cache_vs_quantization",
      "description": "KV cache 与量化是不同优化手段，不应合并",
      "initial_memory": {
        "entries": {
          "model_quantization": {
            "aliases": ["模型量化", "quantization", "INT8", "INT4"],
            "recent_scores": [7],
            "avg_score": 7.0,
            "last_update": "2026-01-01",
            "comments": { "strength": [], "weakness": [] }
          }
        }
      },
      "question": "LLM 推理中的 KV cache 是什么？它如何解决自回归生成的效率问题？",
      "user_answer": "KV cache 在自回归生成时缓存已计算过的 key/value 向量，避免每生成一个新 token 就重新计算整个序列的 attention。这使生成复杂度从 O(n²) 降到 O(n)，大幅加速长序列推理。",
      "llm_reference": "自回归生成中，每个新 token 的 attention 只需与历史 token 的 K/V 交互。KV cache 存储每层已计算的 K/V 矩阵，新 token 只计算自己的 Q/K/V 并与 cache 拼接。代价是显存随序列长度线性增长。PagedAttention（vLLM）通过分页管理 KV cache 提升显存利用率。",
      "scoring_points": [
        { "name": "quant_not_updated", "points": 4, "check": "key_not_updated", "params": { "key": "model_quantization" } },
        { "name": "kv_key", "points": 4, "check": "key_semantic_match", "params": { "keywords": ["kv", "cache", "缓存"] } },
        { "name": "no_broad", "points": 2, "check": "no_broad_keys" }
      ]
    }
  ]
}
```

- [ ] **Step 4: Create `multi_key.json`**

```json
{
  "category": "multi_key",
  "cases": [
    {
      "id": "multi_key_react_agent",
      "description": "ReAct Agent 大问题应拆出 2-3 个记忆点",
      "initial_memory": { "entries": {} },
      "question": "ReAct Agent 的推理循环是怎样的？它如何通过 tool calling 与 observation 协作完成复杂任务？",
      "user_answer": "ReAct 循环是 Thought → Action → Observation：模型先推理下一步（Thought），再调用工具（Action），读取工具返回（Observation），然后继续推理。通过 function calling 声明工具 schema，模型生成调用参数，框架执行后把结果反馈给模型。",
      "llm_reference": "ReAct（Reasoning + Acting）交替生成推理轨迹和工具调用。Thought 步骤让模型显式规划；Action 通过 function calling 调用外部 API；Observation 将工具输出注入上下文，驱动下一轮 Thought。这与纯 Chain-of-Thought 不同，ReAct 能获取实时外部信息。Tool calling 是 Action 的实现机制。",
      "scoring_points": [
        { "name": "update_count", "points": 4, "check": "update_count_range", "params": { "min": 2, "max": 3 } },
        { "name": "distinct_keys", "points": 3, "check": "keys_semantically_distinct", "params": { "min_count": 2 } },
        { "name": "no_broad", "points": 3, "check": "no_broad_keys" }
      ]
    },
    {
      "id": "multi_key_llm_inference",
      "description": "LLM 推理优化大问题应覆盖 KV cache / speculative decoding / batching",
      "initial_memory": { "entries": {} },
      "question": "生产环境中部署 LLM 时，KV cache、投机解码（speculative decoding）和 continuous batching 分别如何提升推理吞吐？",
      "user_answer": "KV cache 避免重复计算历史 token 的 attention，加速自回归生成。投机解码用小模型快速生成候选 token，大模型并行验证，减少大模型前向次数。Continuous batching 在 batch 中动态插入新请求、移除已完成请求，提高 GPU 利用率。",
      "llm_reference": "KV cache 将历史 K/V 缓存，单步生成从 O(n²) 降至 O(n)，代价是显存随序列增长。Speculative decoding 用 draft model 生成 γ 个候选 token，target model 一次前向验证，接受连续匹配的前缀，可提速 2-3x。Continuous batching（vLLM/TGI）不等整个 batch 完成才调度，动态填充空闲 slot，显著提升吞吐。",
      "scoring_points": [
        { "name": "update_count", "points": 3, "check": "update_count_range", "params": { "min": 2, "max": 3 } },
        { "name": "sub_concepts", "points": 4, "check": "keyword_groups_covered", "params": { "groups": [["kv", "cache", "缓存"], ["speculative", "投机"], ["batch", "批"]] } },
        { "name": "has_alias", "points": 3, "check": "alias_nonempty" }
      ]
    }
  ]
}
```

- [ ] **Step 5: Create `scoring.json`**

```json
{
  "category": "scoring",
  "cases": [
    {
      "id": "score_good_decoding",
      "description": "优秀回答应得高分，strength 具体，weakness 为 null",
      "initial_memory": { "entries": {} },
      "question": "LLM 推理时 temperature、top-p（nucleus sampling）和 top-k 采样各自如何影响输出多样性和质量？",
      "user_answer": "Temperature 缩放 logits 分布：越高输出越随机多样，越低越确定保守。Top-k 只从概率最高的 k 个 token 中采样，限制候选范围。Top-p 动态选择累积概率达到 p 的最小 token 集合，自适应候选数量。实际应用中 temperature 和 top-p 常配合使用。",
      "llm_reference": "Temperature T 将 logits 除以 T：T→0 趋近 greedy decoding，T>1 分布更平坦、输出更多样。Top-k 固定候选集大小，可能排除合理低概率 token 或保留不合理高概率 token。Top-p（nucleus sampling）自适应截断，保留累积概率达 p 的最小集合，在多样性和质量间更灵活。生产环境常用 temperature=0.7, top-p=0.9。",
      "scoring_points": [
        { "name": "score_accurate", "points": 4, "check": "score_near_standard", "params": { "standard_score": 8, "threshold": 2 } },
        { "name": "strength_specific", "points": 3, "check": "comment_specific", "params": { "field": "strength" } },
        { "name": "weakness_null", "points": 3, "check": "weakness_is_null" }
      ]
    },
    {
      "id": "score_poor_decoding",
      "description": "差回答应得低分，weakness 具体，strength 为 null",
      "initial_memory": { "entries": {} },
      "question": "LLM 推理时 temperature、top-p（nucleus sampling）和 top-k 采样各自如何影响输出多样性和质量？",
      "user_answer": "temperature 越高越好，top-k 和 top-p 差不多。",
      "llm_reference": "Temperature T 将 logits 除以 T：T→0 趋近 greedy decoding，T>1 分布更平坦、输出更多样。Top-k 固定候选集大小，可能排除合理低概率 token 或保留不合理高概率 token。Top-p（nucleus sampling）自适应截断，保留累积概率达 p 的最小集合，在多样性和质量间更灵活。生产环境常用 temperature=0.7, top-p=0.9。",
      "scoring_points": [
        { "name": "score_accurate", "points": 4, "check": "score_near_standard", "params": { "standard_score": 2, "threshold": 2 } },
        { "name": "weakness_specific", "points": 3, "check": "comment_specific", "params": { "field": "weakness" } },
        { "name": "strength_null", "points": 3, "check": "strength_is_null" }
      ]
    }
  ]
}
```

- [ ] **Step 6: Validate all cases load**

```bash
conda run -n agent python -c "
from pathlib import Path
from tests.benchmarks.memory_loader import load_cases
cases = load_cases(Path('tests/benchmarks/memory/cases'))
print(len(cases), 'cases loaded')
assert len(cases) == 10
for c in cases:
    from tests.benchmarks.memory_scorer import validate_check_names
    validate_check_names(c)
print('OK')
"
```

Expected: `10 cases loaded` / `OK`

---

## Task 6: Integration Benchmark Runner

**Files:**
- Create: `tests/test_memory_benchmark.py`

- [ ] **Step 1: Implement benchmark runner**

```python
"""
Memory Node Benchmark — 真实 LLM 调用，打分评估。

运行方式：
    conda run -n agent pytest tests/test_memory_benchmark.py -v -m integration -s
    conda run -n agent pytest tests/test_memory_benchmark.py -k reuse_text_embedding -m integration -s
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from biteme.config import settings as real_settings
from biteme.graph.memory import apply_updates, invoke_memory_update, load_memory, save_memory
from biteme.graph.prompts import get_memory_prompt, get_prompts
from tests.benchmarks.memory_loader import load_cases, MemoryBenchmarkCase
from tests.benchmarks.memory_scorer import BenchmarkResult, CaseScore, score_case

pytestmark = pytest.mark.integration

CASES_DIR = Path(__file__).parent / "benchmarks" / "memory" / "cases"
ALL_CASES = load_cases(CASES_DIR)

DEFAULT_CONFIG = {"prompt_variant": "default", "model": None}

# Module-level report accumulator (reset per session via fixture)
_report: dict[str, list] = {}


def _make_state(case: MemoryBenchmarkCase) -> dict:
    return {
        "hitl_flags": ["answerer"],
        "messages": [
            {"speaker": "questioner", "content": case.question, "retrieved_chunks": []},
            {"speaker": "human", "content": case.user_answer, "retrieved_chunks": []},
        ],
        "llm_reference_answer": case.llm_reference,
        "mode": "learn",
        "source_path": "/tmp/fake",
    }


def _run_case(case: MemoryBenchmarkCase, config: dict, tmp_path: Path) -> BenchmarkResult:
    memory_path = tmp_path / "memory.json"
    save_memory(case.initial_memory, memory_path)
    data = load_memory(memory_path)
    existing_keys = [
        {"key": k, "aliases": v["aliases"]}
        for k, v in data["entries"].items()
    ]

    variant = config.get("prompt_variant", "default")
    model = config.get("model") or real_settings.openai_model

    def _patched_get_prompts(mode: str) -> dict[str, str]:
        prompts = get_prompts(mode)
        prompts = dict(prompts)
        prompts["memory"] = get_memory_prompt(variant)
        return prompts

    with patch("biteme.graph.memory.get_prompts", _patched_get_prompts):
        invoke_result = invoke_memory_update(
            existing_keys=existing_keys,
            question=case.question,
            user_answer=case.user_answer,
            llm_reference=case.llm_reference,
            mode="learn",
            model=model,
        )

    if invoke_result.parse_ok and invoke_result.updates:
        apply_updates(data, invoke_result.updates)
        save_memory(data, memory_path)

    written = load_memory(memory_path)
    return BenchmarkResult(
        written_memory=written,
        updates=invoke_result.updates,
        parse_ok=invoke_result.parse_ok,
    )


def _format_case_line(scored: CaseScore) -> str:
    if scored.parse_failed:
        return f"{scored.case_id:<35}  0/{scored.max_points}  [PARSE FAILED]"
    marks = "".join("✓" if p.passed else "✗" for p in scored.points)
    return f"{scored.case_id:<35} {scored.earned:>2}/{scored.max_points}  {marks}"


@pytest.fixture(scope="session", autouse=True)
def _benchmark_report(request):
    _report.clear()
    yield
    # Print aggregate after all tests in this file
    for config_id, scores in _report.items():
        total_earned = sum(s.earned for s in scores)
        total_max = sum(s.max_points for s in scores)
        failures = [s.case_id for s in scores if s.parse_failed]
        print(f"\n=== Memory Benchmark [{config_id}] ===")
        for s in scores:
            print(_format_case_line(s))
        print("-" * 50)
        pct = 100.0 * total_earned / total_max if total_max else 0
        print(f"Total: {total_earned}/{total_max} ({pct:.1f}%)")
        if failures:
            print(f"Parse failures ({len(failures)}): {', '.join(failures)}")


@pytest.mark.parametrize("case", ALL_CASES, ids=[c.id for c in ALL_CASES])
@pytest.mark.parametrize("config", [DEFAULT_CONFIG], ids=["default"])
def test_memory_benchmark_case(case, config, tmp_path, request):
    config_id = request.node.callspec.id.split("-")[0] if hasattr(request.node, "callspec") else "default"
    result = _run_case(case, config, tmp_path)
    scored = score_case(case, result)
    _report.setdefault(config_id, []).append(scored)
    # Observational — always pass; scores printed in session fixture
    assert scored.max_points == 10
```

- [ ] **Step 2: Run loader unit tests (sanity)**

```bash
conda run -n agent pytest tests/test_memory_benchmark_loader.py -v
```

Expected: PASS

- [ ] **Step 3: Run one integration case (requires OPENAI_API_KEY)**

```bash
conda run -n agent pytest tests/test_memory_benchmark.py -k new_key_langgraph_checkpointer -m integration -s
```

Expected: test PASS (observational), prints score line and session summary

- [ ] **Step 4: Run full benchmark (optional, ~10 LLM calls)**

```bash
conda run -n agent pytest tests/test_memory_benchmark.py -m integration -s
```

Expected: 10 tests PASS, session summary with Total: X/100

---

## Task 7: Update Spec Status

**Files:**
- Modify: `docs/superpowers/specs/2026-06-09-memory-benchmark-design.md`

- [ ] **Step 1: Set spec status to Approved**

Change line 4 from `Draft — pending user review` to `Approved`.

---

## Self-Review (spec coverage)

| Spec requirement | Task |
|------------------|------|
| Per-category JSON files | Task 5 |
| Loader flatten + validation | Task 1 |
| All check functions | Task 3 |
| Standard score + threshold | Task 3 `_check_score_near_standard` |
| Parse failure → 0 + track list | Task 3 + Task 6 session report |
| `invoke_memory_update` extraction | Task 2 |
| Prompt variants | Task 4 |
| pytest integration + parametrize | Task 6 |
| Keep `test_memory_eval.py` | No changes |
| 10 cases / 100 points | Task 5 |

---

## Local A/B Comparison (document in test file comment)

To compare prompt variants locally, change Task 6 parametrize:

```python
@pytest.mark.parametrize("config", [
    {"prompt_variant": "default", "model": None},
    # {"prompt_variant": "v2", "model": "glm-5"},
], ids=["default", "v2"])
```

Add new variants to `MEMORY_PROMPT_VARIANTS` in `prompts.py`.
