from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from .state import SessionState, Turn
from .prompts import get_prompts
from ..context.factory import create_provider
from ..config import settings


def _get_db_path(source_path: str) -> str:
    import hashlib
    h = hashlib.md5(source_path.encode()).hexdigest()[:8]
    return str(settings.biteme_home / "indexes" / h)


def questioner_node(state: SessionState) -> dict:
    if "questioner" in state["hitl_flags"]:
        human_text = interrupt(">>> [提问者] 请输入你的问题：")
        turn: Turn = {"speaker": "human", "content": human_text, "retrieved_chunks": []}
    else:
        prompts = get_prompts(state["mode"])
        history = "\n".join(
            f"[{t['speaker']}]: {t['content']}" for t in state["messages"][-6:]
        )
        provider = create_provider(
            source_path=state["source_path"],
            strategy=state["context_strategy"],
            db_path=_get_db_path(state["source_path"]),
        )
        # First turn: use get_overview() so the questioner can survey source content
        # Subsequent turns: retrieve relevant chunks using the last answer as query
        if not state["messages"]:
            context_chunks = provider.get_overview()
        else:
            retrieval_query = state["messages"][-1]["content"]
            context_chunks = provider.retrieve(retrieval_query)
        context_text = "\n\n---\n\n".join(context_chunks[:3])
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        response = llm.invoke([
            SystemMessage(content=prompts["questioner"]),
            HumanMessage(content=f"对话历史：\n{history}\n\n参考内容摘要：\n{context_text[:2000]}\n\n请提出下一个问题。"),
        ])
        turn = {"speaker": "questioner", "content": response.content, "retrieved_chunks": []}

    return {
        "messages": state["messages"] + [turn],
        "current_speaker": "answerer",
        "turn_count": state["turn_count"] + 1,
    }


def answerer_node(state: SessionState) -> dict:
    last_question = state["messages"][-1]["content"] if state["messages"] else ""
    provider = create_provider(
        source_path=state["source_path"],
        strategy=state["context_strategy"],
        db_path=_get_db_path(state["source_path"]),
    )
    chunks = provider.retrieve(last_question)  # always called, never skipped

    if "answerer" in state["hitl_flags"]:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel("\n".join(chunks[:3]), title="[yellow]检索到的相关内容[/yellow]"))
        human_text = interrupt(">>> [回答者] 请输入你的回答：")
        turn: Turn = {"speaker": "human", "content": human_text, "retrieved_chunks": chunks}
    else:
        prompts = get_prompts(state["mode"])
        context_text = "\n\n---\n\n".join(chunks[:5])
        answerer_prompt = prompts["answerer"].format(context=context_text)

        history = "\n".join(
            f"[{t['speaker']}]: {t['content']}" for t in state["messages"][-6:]
        )
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=answerer_prompt),
            HumanMessage(content=f"对话历史：\n{history}\n\n请回答最后那个问题。"),
        ])
        turn = {"speaker": "answerer", "content": response.content, "retrieved_chunks": chunks}

    return {
        "messages": state["messages"] + [turn],
        "current_speaker": "questioner",
    }
