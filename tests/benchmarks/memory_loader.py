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
