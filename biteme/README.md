# BiteMe

LangGraph-based dual-agent project understanding assistant.

## Quick Start

### 1) Install dependencies

```bash
cd biteme
python -m pip install -e .
```

### 2) Run the CLI

```bash
python -m src.cli
```

The CLI collects a short project summary and optional comma-separated keywords,
then runs the orchestrator graph end-to-end and prints the final session summary.

CLI usage flow:
- Choose `observe` (default) or `interview` mode.
- Optionally queue intervention commands before execution:
  - `/ask <topic>` inserts a high-priority question.
  - `/skip` skips the next pending question.
  - `/stop` terminates the session early.
- In `interview` mode, provide an optional answer for the current question; this
  is captured into the orchestrator state and evaluated with the current
  placeholder signal.

### 3) Run tests

```bash
pytest -q
```

Run only orchestrator integration tests:

```bash
pytest tests/test_orchestrator_flow.py -q
```

## Current Scope Notes

- Interview-mode answer evaluation is currently a placeholder signal (`placeholder-score`), not a production grading rubric.
- Private per-agent memory (`src/memory/private.py`) exists as scaffolding but is not integrated into orchestrator flow yet.
