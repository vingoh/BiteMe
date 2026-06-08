from __future__ import annotations

import json
import logging
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
from .prompts import get_prompts
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

    prompts = get_prompts(state["mode"])
    prompt_text = prompts["memory"].format(
        existing_keys=existing_keys,
        question=question,
        user_answer=user_answer,
        llm_reference=llm_reference,
    )

    llm = ChatOpenAI(model=settings.openai_model, temperature=0.0)
    structured_llm = llm.with_structured_output(MemoryUpdates)

    result: MemoryUpdates | None = None
    try:
        result = structured_llm.invoke([HumanMessage(content=prompt_text)])
    except Exception as exc:
        # Fallback: models without native structured output still return valid
        # JSON in message.content (often markdown-wrapped). Recover it instead
        # of skipping the memory update entirely.
        raw_content: str | None = None
        if isinstance(exc, ValidationError):
            # Reuse the LLM response already embedded in the parse error — no
            # extra API call needed.
            for err in exc.errors():
                if err.get("type") == "json_invalid" and isinstance(err.get("input"), str):
                    raw_content = err["input"]
                    break

        if raw_content is None:
            # Last resort when the error carries no parseable input.
            try:
                raw_resp = llm.invoke([HumanMessage(content=prompt_text)])
                raw_content = raw_resp.content
            except Exception:
                logger.exception("memory_node LLM call failed — skipping memory update")
                return {}
        try:
            result = _parse_memory_updates(raw_content)
        except Exception:
            logger.exception("memory_node LLM call failed — skipping memory update")
            return {}

    apply_updates(data, result.updates)
    save_memory(data, memory_path)

    console = Console()
    summary = "\n".join(
        f"  [{u.key}] score={u.score}  strength: {u.strength}  weakness: {u.weakness}"
        for u in result.updates
    )
    console.print(Panel(summary, title="[magenta]Memory Updated[/magenta]"))

    return {}
