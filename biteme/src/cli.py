from typing import Optional

try:
    from rich.console import Console
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight local envs
    class Console:  # type: ignore[override]
        def print(self, message) -> None:
            print(message)

from src.memory.shared import GraphState
from src.memory.shared import new_graph_state
from src.orchestrator import apply_user_intervention, build_graph, run_session
from src.agents.summarizer import summarizer_node


def run_cli_session(
    *,
    project_content: str,
    user_keywords: list[str],
    mode: str = "observe",
    max_rounds: int = 8,
    intervention_commands: Optional[list[str]] = None,
    interview_answer: Optional[str] = None,
) -> GraphState:
    state = new_graph_state(project_content=project_content, mode=mode, max_rounds=max_rounds)
    state["user_keywords"] = user_keywords

    for command in intervention_commands or []:
        apply_user_intervention(state, command)

    if mode == "interview" and interview_answer is not None:
        state["user_answer"] = interview_answer

    if state["terminated"]:
        return summarizer_node(state, llm=None)

    graph = build_graph()
    return run_session(graph, state)


def run_cli() -> None:
    console = Console()
    project_content = input("Project content summary: ").strip()
    keyword_input = input("Keywords (comma separated, optional): ").strip()
    keywords = [item.strip() for item in keyword_input.split(",") if item.strip()]
    mode_input = input("Mode (observe/interview, default observe): ").strip().lower()
    mode = mode_input if mode_input in {"observe", "interview"} else "observe"

    intervention_commands: list[str] = []
    while True:
        command = input("Intervention (/ask topic, /skip, /stop, Enter to continue): ").strip()
        if not command:
            break
        if command.startswith("/ask ") or command.startswith("/skip") or command.startswith("/stop"):
            intervention_commands.append(command)
            if command.startswith("/stop"):
                break
            continue
        console.print("Ignored unsupported command. Use /ask, /skip, or /stop.")

    interview_answer = None
    if mode == "interview":
        interview_answer = input("Interview answer for current question (optional): ").strip()

    result = run_cli_session(
        project_content=project_content,
        user_keywords=keywords,
        mode=mode,
        intervention_commands=intervention_commands,
        interview_answer=interview_answer,
    )

    console.print("BiteMe CLI started.")
    console.print(f"Session finished. reason={result.get('termination_reason', 'unknown')}")
    console.print(f"Completed questions: {len(result['completed_question_ids'])}")
    console.print(f"Summary: {result.get('final_summary', {})}")


if __name__ == "__main__":
    run_cli()
