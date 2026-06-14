from __future__ import annotations

import json
import logging
from dataclasses import dataclass
import os
import tempfile
from datetime import date
from pathlib import Path
from statistics import mean
from typing import TypedDict

from pydantic import BaseModel, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel

from ..config import settings
from .prompts import get_prompts, MEMORY_RECALL_PROMPT, MEMORY_REFINE_PROMPT
from .state import SessionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TypedDicts for the in-memory representation of memory.json
# ---------------------------------------------------------------------------

class MemoryComments(TypedDict):
    strength: list[str]
    weakness: list[str]


class MemoryEntry(TypedDict):
    aliases: list[str]
    recent_scores: list[int]
    avg_score: float
    last_update: str          # YYYY-MM-DD
    comments: MemoryComments


class MemoryFile(TypedDict):
    entries: dict[str, MemoryEntry]


# ---------------------------------------------------------------------------
# Pydantic schemas for LLM structured output
# ---------------------------------------------------------------------------

class MemoryUpdate(BaseModel):
    key: str
    aliases: list[str]        # if key is new: initial aliases; if key exists: new aliases to add (may be empty)
    score: int                # 0-10
    strength: str | None      # specific strength; null if none identifiable
    weakness: str | None      # specific error/omission; null if none identifiable


class MemoryUpdates(BaseModel):
    updates: list[MemoryUpdate]


class RecalledEntry(BaseModel):
    key: str
    relevance_reason: str  # must cite alias or comment text as evidence


class MemoryRecallResult(BaseModel):
    recalled: list[RecalledEntry]  # max 3


# ---------------------------------------------------------------------------
# LLM response parsing (fallback for models without native structured output)
# ---------------------------------------------------------------------------

def _strip_markdown_json(content: str) -> str:
    """Remove optional markdown code fences around JSON content.

    Some models (e.g. glm-5.1) return ```json ... ``` instead of raw JSON,
    which breaks with_structured_output parsing.
    """
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[-1].strip() == "```":
        lines = lines[1:-1]
    elif lines:
        lines = lines[1:]
    return "\n".join(lines).strip()


def _parse_memory_updates(content: str) -> MemoryUpdates:
    """Parse LLM text into MemoryUpdates, tolerating markdown-wrapped JSON."""
    cleaned = _strip_markdown_json(content)
    try:
        return MemoryUpdates.model_validate_json(cleaned)
    except Exception:
        # Some models return a bare array [{...}] instead of {"updates": [...]}.
        payload = json.loads(cleaned)
        if isinstance(payload, list):
            payload = {"updates": payload}
        return MemoryUpdates.model_validate(payload)


def _format_memory_entries_for_recall(memory_data: MemoryFile) -> str:
    """Format all memory entries as a string for the recall prompt."""
    lines = []
    for key, entry in memory_data["entries"].items():
        aliases = entry["aliases"]
        strengths = entry["comments"]["strength"][-3:]
        weaknesses = entry["comments"]["weakness"][-3:]
        lines.append(
            f"key: {key}\n"
            f"  aliases: {aliases}\n"
            f"  comments.strength: {strengths}\n"
            f"  comments.weakness: {weaknesses}"
        )
    return "\n\n".join(lines)


def _format_recalled_entries_for_refine(
    recalled: list[RecalledEntry],
    memory_data: MemoryFile,
) -> str:
    """Format recalled entries with scores/dates/comments for the refine prompt."""
    lines = []
    for entry in recalled:
        mem = memory_data["entries"].get(entry.key)
        if mem is None:
            continue
        strengths = mem["comments"]["strength"][-3:]
        weaknesses = mem["comments"]["weakness"][-3:]
        lines.append(
            f"key: {entry.key}\n"
            f"  avg_score: {mem['avg_score']}\n"
            f"  last_update: {mem['last_update']}\n"
            f"  relevance_reason: {entry.relevance_reason}\n"
            f"  comments.strength: {strengths}\n"
            f"  comments.weakness: {weaknesses}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_memory(path: Path) -> MemoryFile:
    """Load memory.json; return empty structure if file is absent or corrupt."""
    if not path.exists():
        return {"entries": {}}
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            logger.warning("memory.json is corrupt — starting fresh")
            return {"entries": {}}


def save_memory(data: MemoryFile, path: Path) -> None:
    """Atomically write memory data to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def apply_updates(data: MemoryFile, updates: list[MemoryUpdate]) -> None:
    """Apply LLM-produced updates to the in-memory data dict (mutates in place)."""
    today = date.today().isoformat()
    for update in updates:
        if update.key not in data["entries"]:
            data["entries"][update.key] = {
                "aliases": [],          # populated by the merge loop below
                "recent_scores": [],
                "avg_score": 0.0,
                "last_update": today,
                "comments": {"strength": [], "weakness": []},
            }
        entry = data["entries"][update.key]

        # Merge aliases (order-preserving dedup) — handles both new and existing entries
        existing_set: set[str] = set(entry["aliases"])
        for alias in update.aliases:
            if alias not in existing_set:
                entry["aliases"].append(alias)
                existing_set.add(alias)

        entry["recent_scores"].append(update.score)
        entry["recent_scores"] = entry["recent_scores"][-10:]
        entry["avg_score"] = mean(entry["recent_scores"])
        entry["last_update"] = today
        if update.strength is not None:
            entry["comments"]["strength"].append(update.strength)
        if update.weakness is not None:
            entry["comments"]["weakness"].append(update.weakness)


# ---------------------------------------------------------------------------
# LLM invoke (no disk I/O)
# ---------------------------------------------------------------------------

def recall_memory(
    draft_question: str,
    memory_data: MemoryFile,
    model: str | None = None,
) -> list[RecalledEntry]:
    """Call LLM to find top-3 most relevant memory entries for the draft question.

    Returns empty list if memory is empty, draft is empty, or LLM call fails.
    """
    if not draft_question.strip():
        return []
    if not memory_data["entries"]:
        return []

    memory_entries_text = _format_memory_entries_for_recall(memory_data)
    prompt_text = MEMORY_RECALL_PROMPT.format(
        draft_question=draft_question,
        memory_entries=memory_entries_text,
    )

    llm_model = model or settings.openai_model
    llm = ChatOpenAI(model=llm_model, temperature=0.0)
    structured_llm = llm.with_structured_output(MemoryRecallResult)

    try:
        result: MemoryRecallResult = structured_llm.invoke([HumanMessage(content=prompt_text)])
        valid_entries = [
            e for e in result.recalled
            if e.key in memory_data["entries"]
        ]
        return valid_entries[:3]
    except Exception:
        logger.warning("recall_memory LLM call failed", exc_info=True)
        return []


def refine_question(
    draft_question: str,
    recalled: list[RecalledEntry],
    memory_data: MemoryFile,
    model: str | None = None,
) -> str:
    """Refine draft_question based on recalled memory entries.

    Returns draft_question unchanged if LLM fails or returns empty string.
    """
    recalled_entries_text = _format_recalled_entries_for_refine(recalled, memory_data)
    prompt_text = MEMORY_REFINE_PROMPT.format(
        draft_question=draft_question,
        recalled_entries=recalled_entries_text,
    )

    llm_model = model or settings.openai_model
    llm = ChatOpenAI(model=llm_model, temperature=0.7)

    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        refined = response.content.strip()
        if not refined:
            return draft_question
        return refined
    except Exception:
        logger.warning("refine_question LLM call failed", exc_info=True)
        return draft_question


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


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def memory_node(state: SessionState) -> dict:
    if "answerer" not in state["hitl_flags"]:
        return {}

    messages = state["messages"]
    if len(messages) < 2:
        return {}

    question = messages[-2]["content"]
    user_answer = messages[-1]["content"]
    llm_reference = state.get("llm_reference_answer", "")

    memory_path = settings.biteme_home / "memory.json"
    data = load_memory(memory_path)

    existing_keys = [
        {"key": k, "aliases": v["aliases"]}
        for k, v in data["entries"].items()
    ]

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
