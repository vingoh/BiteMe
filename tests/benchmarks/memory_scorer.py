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
