from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console

_console = Console()


def stream_agent(agent, input_messages: list, *, recursion_limit: int = 12, verbose: bool = True) -> str:
    """Run a ReAct agent with optional step-by-step logging. Returns final answer text."""
    final_content = ""

    for chunk in agent.stream(
        {"messages": input_messages},
        recursion_limit=recursion_limit,
    ):
        print(chunk)
        for _node_name, node_output in chunk.items():
            for msg in node_output.get("messages", []):
                if verbose:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        if msg.content:
                            _console.print(f"[dim]💭 {msg.content}[/dim]")
                        for tc in msg.tool_calls:
                            _console.print(
                                f"[cyan]🔧 工具调用: {tc['name']}[/cyan]  参数: {tc['args']}"
                            )
                    elif isinstance(msg, ToolMessage):
                        _console.print(
                            f"[green]📄 工具结果 ({msg.name}):[/green] {str(msg.content)[:500]}"
                        )

                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_content = msg.content

    return final_content
