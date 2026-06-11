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
from tests.benchmarks.memory_scorer import BenchmarkResult, CaseScore, score_case, validate_check_names

pytestmark = pytest.mark.integration

CASES_DIR = Path(__file__).parent / "benchmarks" / "memory" / "cases"
ALL_CASES = load_cases(CASES_DIR)
for _case in ALL_CASES:
    validate_check_names(_case)

DEFAULT_CONFIG = {"prompt_variant": "default", "model": None}

# Module-level report accumulator (reset per session via fixture)
_report: dict[str, list] = {}


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
    marks = " ".join(
        f"{p.name}{'✓' if p.passed else '✗'}" for p in scored.points
    )
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
def test_memory_benchmark_case(case, config, tmp_path):
    config_id = config.get("prompt_variant", "default")
    result = _run_case(case, config, tmp_path)
    scored = score_case(case, result)
    _report.setdefault(config_id, []).append(scored)
    # Observational — always pass; scores printed in session fixture
    assert scored.max_points == 10
