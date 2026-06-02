import os
import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import interrupt
from rich.console import Console
from rich.panel import Panel
from .state import SessionState, Turn
from .prompts import get_prompts, MEMORY_MERGE

from langchain.agents import create_agent
from ..context.factory import create_provider
from ..config import settings
from ..tools import READONLY_TOOLS
from .agent_runner import stream_agent

_MAX_MEMORY_RETRIES = 3


def _get_db_path(source_path: str) -> str:
    import hashlib
    h = hashlib.md5(source_path.encode()).hexdigest()[:8]
    return str(settings.biteme_home / "indexes" / h)


def _parse_outline(text: str) -> list[str]:
    """Parse numbered list from LLM output into plain question strings."""
    questions = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
        if cleaned:
            questions.append(cleaned)
    return questions


def planner_node(state: SessionState) -> dict:
    prompts = get_prompts(state["mode"])
    n = state["max_turns"] + 2

    console = Console()
    console.print("\n[bold cyan][PLANNER][/bold cyan]")
    llm = ChatOpenAI(model=settings.openai_model, temperature=0.7)
    planner_agent = create_agent(
        model=llm,
        tools=READONLY_TOOLS,
        system_prompt=prompts["planner"],
    )
    outline_text = stream_agent(
        planner_agent,
        [HumanMessage(
            content=(
                f"源文件路径：{state['source_path']}\n\n"
                f"请先探索项目结构，再生成 {n} 个问题，按编号列出。"
            )
        )],
    )

    outline = _parse_outline(outline_text)

    title = "提问大纲" if state["mode"] == "learn" else "面试大纲"
    outline_display = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(outline))
    console.print(Panel(outline_display, title=f"[cyan]{title}[/cyan]"))

    return {"outline": outline}


def questioner_node(state: SessionState) -> dict:
    if "questioner" in state["hitl_flags"]:
        Console().print("\n[bold blue][USER QUESTIONER][/bold blue]")
        outline = state.get("outline", [])
        turn_idx = state["turn_count"]
        if outline and turn_idx < len(outline):
            suggested = outline[turn_idx]
            prompt_msg = (
                f"建议问题（第 {turn_idx + 1} 轮）：{suggested}\n"
                f"（直接回车使用建议问题，或输入新问题）"
            )
        else:
            prompt_msg = "请输入你的问题："
        human_text = interrupt(prompt_msg)
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

        outline = state.get("outline", [])
        outline_section = ""
        turn_idx = state["turn_count"]
        if outline and turn_idx < len(outline):
            current_topic = outline[turn_idx]
            outline_section = (
                f"\n\n本轮话题方向（仅供参考，请结合上一轮回答内容自由发挥，"
                f"不要照搬原文，提出你自己的问题）：{current_topic}"
            )

        console = Console()
        console.print("\n[bold blue][QUESTIONER][/bold blue]")
        llm = ChatOpenAI(model=settings.openai_model, temperature=0.7)
        questioner_agent = create_agent(
            model=llm,
            tools=READONLY_TOOLS,
            system_prompt=prompts["questioner"],
        )
        question_text = stream_agent(
            questioner_agent,
            [HumanMessage(
                content=(
                    f"对话历史：\n{history}"
                    f"{outline_section}"
                    f"\n\n参考内容摘要：\n{context_text[:2000]}"
                    f"\n\n请提出下一个问题。"
                )
            )],
        )
        turn = {"speaker": "questioner", "content": question_text, "retrieved_chunks": []}

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
        import click
        console = Console()
        console.print("\n[bold green][USER ANSWERER][/bold green]")

        

        def _make_preview(chunks: list[str], max_lines: int = 3, max_chars: int = 200) -> str:
            if not chunks:
                return ""
            first = chunks[0]
            lines = first.splitlines()
            preview = "\n".join(lines[:max_lines])
            if len(preview) > max_chars:
                preview = preview[:max_chars]
            truncated = len(lines) > max_lines or len(first) > max_chars or len(chunks) > 1
            return preview + (" …" if truncated else "")

        preview_text = _make_preview(chunks)
        console.print(Panel(
            preview_text,
            title="[yellow]检索到的相关内容[/yellow] [dim]▸ 折叠[/dim]",
        ))
        console.print("[dim]按 [bold]e[/bold] 展开全部内容，按任意键继续作答[/dim]")
        key = click.getchar()
        if key.lower() == "e":
            console.print(Panel(
                "\n\n─────\n\n".join(chunks[:3]),
                title="[yellow]检索到的相关内容[/yellow] [dim]▾ 展开[/dim]",
            ))

        human_text = interrupt("请输入你的回答：")

        prompts = get_prompts(state["mode"])
        context_text = "\n\n---\n\n".join(chunks[:5])
        history = "\n".join(
            f"[{t['speaker']}]: {t['content']}" for t in state["messages"][-6:]
        )
        llm_ref = ChatOpenAI(model=settings.openai_model, temperature=0.3)
        ref_agent = create_agent(
            model=llm_ref,
            tools=READONLY_TOOLS,
            system_prompt=prompts["answerer"],
        )
        # 用户提交答案后，调用 LLM 生成参考答案并展示
        console.print("\n[bold green][LLM 参考答案生成中][/bold green]")
        llm_answer = stream_agent(
            ref_agent,
            [HumanMessage(
                content=(
                    f"源文件路径：{state['source_path']}"
                    f"检索到的相关内容：\n{context_text}"
                    f"\n\n对话历史：\n{history}"
                    f"\n\n请回答最后那个问题。"
                )
            )],
        )
        console.print(Panel(
            llm_answer,
            title="[green]LLM 参考答案[/green]",
        ))

        turn: Turn = {"speaker": "human", "content": human_text, "retrieved_chunks": chunks}
        return {
            "messages": state["messages"] + [turn],
            "current_speaker": "questioner",
            "llm_reference_answer": llm_answer,
        }
    else:
        console = Console()
        console.print("\n[bold green][ANSWERER][/bold green]")
        prompts = get_prompts(state["mode"])
        context_text = "\n\n---\n\n".join(chunks[:5])

        history = "\n".join(
            f"[{t['speaker']}]: {t['content']}" for t in state["messages"][-6:]
        )
        llm = ChatOpenAI(model=settings.openai_model, temperature=0.3)
        react_agent = create_agent(
            model=llm,
            tools=READONLY_TOOLS,
            system_prompt=prompts["answerer"],
        )
        final_answer = stream_agent(
            react_agent,
            [HumanMessage(
                content=(
                    f"源文件路径：{state['source_path']}"
                    f"检索到的相关内容：\n{context_text}"
                    f"\n\n对话历史：\n{history}"
                    f"\n\n请回答最后那个问题。"
                )
            )],
        )
        turn = {"speaker": "answerer", "content": final_answer, "retrieved_chunks": chunks}
        return {
            "messages": state["messages"] + [turn],
            "current_speaker": "questioner",
            "llm_reference_answer": "",
        }


def reviewer_node(state: SessionState) -> dict:
    question = state["messages"][-2]["content"] if len(state["messages"]) >= 2 else ""
    user_answer = state["messages"][-1]["content"] if state["messages"] else ""
    llm_answer = state.get("llm_reference_answer", "")

    prompts = get_prompts(state["mode"])
    llm = ChatOpenAI(model=settings.openai_model, temperature=0.1)

    response = llm.invoke([
        SystemMessage(content=prompts["reviewer"]),
        HumanMessage(content=(
            f"问题：{question}\n\n"
            f"用户回答：{user_answer}\n\n"
            f"LLM 参考答案：{llm_answer}"
        )),
    ])

    try:
        data = json.loads(response.content)
        keywords = [
            k for k in data.get("keywords", [])
            if isinstance(k, dict)
            and isinstance(k.get("keyword"), str)
            and isinstance(k.get("score"), int)
            and 0 <= k["score"] <= 10
        ]
    except (json.JSONDecodeError, AttributeError, TypeError):
        keywords = []

    console = Console()

    def _color(score: int) -> str:
        if score >= 7:
            return "green"
        if score >= 4:
            return "yellow"
        return "red"

    lines = "\n".join(
        f"[{_color(kw['score'])}]{kw['keyword']}[/{_color(kw['score'])}]  {kw['score']}"
        for kw in keywords
    )
    console.print(Panel(lines or "[dim]（无关键词）[/dim]", title="[bold]本轮评审[/bold]"))

    return {
        "review_history": state["review_history"] + [keywords],
    }


def memory_node(state: SessionState) -> dict:
    if not state["review_history"]:
        return {}

    memory_path = settings.review_memory_path
    old_memory: list = []
    if memory_path.exists():
        try:
            old_memory = json.loads(memory_path.read_text(encoding="utf-8"))
        except Exception:
            Console().print("[yellow]警告：无法读取 review_memory.json，视为空记忆[/yellow]")
            old_memory = []

    old_memory_text = json.dumps(old_memory, ensure_ascii=False)
    review_history_text = json.dumps(state["review_history"], ensure_ascii=False)

    llm = ChatOpenAI(model=settings.openai_model, temperature=0.0)
    messages = [
        SystemMessage(content=MEMORY_MERGE),
        HumanMessage(content=(
            f"旧记忆：\n{old_memory_text}\n\n"
            f"本次 session 的 review_history：\n{review_history_text}"
        )),
    ]

    validated: list | None = None
    for attempt in range(_MAX_MEMORY_RETRIES):
        response = llm.invoke(messages)
        raw: str = response.content
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("顶层结构必须是 JSON 数组")
            entries = []
            for entry in data:
                kw = entry.get("keyword")
                scores = entry.get("scores")
                if (
                    isinstance(kw, str) and kw
                    and isinstance(scores, list)
                    and scores
                    and all(isinstance(s, int) and 0 <= s <= 10 for s in scores)
                ):
                    entries.append({"keyword": kw, "scores": scores})
            if not entries:
                raise ValueError("所有条目均未通过验证，视为无效响应")
            validated = entries
            break
        except Exception as exc:
            if attempt < _MAX_MEMORY_RETRIES - 1:
                messages.append(AIMessage(content=raw))
                messages.append(HumanMessage(content=(
                    f"上面的输出有误：{exc}。"
                    f"请修正并重新输出合法 JSON 数组，"
                    f'格式：[{{"keyword": "...", "scores": [...]}}]'
                )))
            else:
                Console().print("[yellow]警告：memory_node 连续 3 次输出无效，跳过写入[/yellow]")
                return {}

    result = [
        {
            "keyword": entry["keyword"],
            "scores": entry["scores"],
            "avg_score": round(sum(entry["scores"]) / len(entry["scores"]), 2),
        }
        for entry in validated
    ]

    try:
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = memory_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, memory_path)
    except Exception:
        Console().print("[yellow]警告：review_memory.json 写入失败[/yellow]")

    return {}
