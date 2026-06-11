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


from biteme.graph.memory import MemoryUpdate
from tests.benchmarks.memory_scorer import BenchmarkResult, score_case, validate_check_names
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


def test_validate_check_names_rejects_unknown():
    case = _case(scoring_points=[ScoringPoint("bad", 10, "nonexistent_check")])
    with pytest.raises(ValueError, match="unknown check"):
        validate_check_names(case)


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
