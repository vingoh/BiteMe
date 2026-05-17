from pathlib import Path
from typing import Annotated
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from langgraph.types import Command

from .config import settings
from .indexing.pipeline import build_index
from .session.manager import create_session, list_sessions, get_checkpoint_saver
from .graph.graph import build_graph
from .graph.state import SessionState, Turn

app = typer.Typer(name="biteme", help="双 Agent 问答系统 CLI")
console = Console()


@app.command()
def index(
    source: Annotated[str, typer.Argument(help="本地目录或文件路径")],
):
    """预建 RAG 索引（大型代码仓或多文档集合时使用）"""
    path = Path(source)
    if not path.exists():
        console.print(f"[red]错误：路径不存在：{source}[/red]")
        raise typer.Exit(code=1)
    import hashlib
    h = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
    db_path = str(settings.biteme_home / "indexes" / h)
    settings.ensure_dirs()
    with console.status("[bold green]正在建索引…"):
        build_index(source_path=str(path.resolve()), db_path=db_path)
    console.print(f"[green]✓ 索引已保存至 {db_path}[/green]")


@app.command()
def run(
    source: Annotated[str, typer.Argument(help="本地目录或文件路径")],
    mode: Annotated[str, typer.Option(help="learn 或 interview")] = "learn",
    hitl: Annotated[str, typer.Option(help="none/questioner/answerer/both")] = "none",
    strategy: Annotated[str, typer.Option(help="auto/direct/rag")] = "auto",
    turns: Annotated[int, typer.Option(help="最大对话轮数")] = 5,
):
    """启动新的双 Agent 对话会话"""
    path = Path(source)
    if not path.exists():
        console.print(f"[red]错误：路径不存在：{source}[/red]")
        raise typer.Exit(code=1)

    hitl_flags = []
    if hitl == "questioner":
        hitl_flags = ["questioner"]
    elif hitl == "answerer":
        hitl_flags = ["answerer"]
    elif hitl == "both":
        hitl_flags = ["questioner", "answerer"]

    settings.ensure_dirs()
    session_id = create_session(source_path=str(path.resolve()), mode=mode)
    console.print(Panel(
        f"会话 ID: [bold]{session_id}[/bold]  模式: [cyan]{mode}[/cyan]  HITL: [yellow]{hitl}[/yellow]",
        title="BiteMe",
    ))

    initial_state: SessionState = {
        "mode": mode,  # type: ignore
        "messages": [],
        "current_speaker": "questioner",
        "hitl_flags": hitl_flags,
        "turn_count": 0,
        "max_turns": turns,
        "context_strategy": strategy,
        "source_path": str(path.resolve()),
        "outline": [],
    }

    with get_checkpoint_saver(session_id) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": session_id}}
        _run_graph(graph, initial_state, config)


@app.command()
def resume(
    session_id: Annotated[str, typer.Argument(help="要恢复的会话 ID")],
):
    """恢复上次中断的会话"""
    with get_checkpoint_saver(session_id) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": session_id}}
        _run_graph(graph, None, config)


@app.command("list")
def list_cmd():
    """列出历史会话"""
    sessions = list_sessions()
    if not sessions:
        console.print("[dim]暂无历史会话[/dim]")
        return
    table = Table(title="历史会话")
    table.add_column("ID")
    table.add_column("来源")
    table.add_column("模式")
    table.add_column("时间")
    table.add_column("状态")
    for s in sessions:
        table.add_row(
            s["session_id"], s["source_path"], s["mode"],
            s["created_at"][:19], s["status"],
        )
    console.print(table)


def _extract_suggestion(prompt_text: str) -> str:
    """从 HITL 提示文字中提取建议问题。

    提示格式：'>>> [提问者] 建议问题（第 N 轮）：{question}\\n（直接回车使用建议问题，或输入新问题）'
    """
    if "建议问题" in prompt_text and "：" in prompt_text:
        parts = prompt_text.split("：", 1)
        if len(parts) > 1:
            suggestion = parts[1].split("\n")[0].strip()
            return suggestion
    return ""


def _print_turn(turn: Turn) -> None:
    """Print a single conversation turn."""
    speaker_color = {
        "questioner": "blue",
        "answerer": "green",
        "human": "yellow",
    }.get(turn["speaker"], "white")
    console.print(
        f"\n[bold {speaker_color}][{turn['speaker'].upper()}][/bold {speaker_color}]"
    )
    console.print(turn["content"])
    if turn.get("retrieved_chunks"):
        console.print(f"[dim]（引用了 {len(turn['retrieved_chunks'])} 个片段）[/dim]")


def _run_graph(graph, initial_state, config):
    """运行图，循环处理 HITL interrupt 与流式输出。"""
    pending = initial_state if initial_state is not None else None
    printed_count = 0

    while True:
        stream = graph.stream(pending, config=config, stream_mode="values")

        interrupted = False
        for state in stream:
            if not isinstance(state, dict):
                continue
            # print(f"state: {state}")
            msgs = state.get("messages") or []
            while printed_count < len(msgs):
                _print_turn(msgs[printed_count])
                printed_count += 1

            if "__interrupt__" in state:
                interrupt_obj = state["__interrupt__"][0]
                prompt_text = str(interrupt_obj.value)
                console.print(f"\n[bold yellow]{prompt_text}[/bold yellow]")
                user_input = typer.prompt("", prompt_suffix="> ")
                if not user_input.strip():
                    user_input = _extract_suggestion(prompt_text)
                pending = Command(resume=user_input)
                interrupted = True
                break

        if not interrupted:
            break
