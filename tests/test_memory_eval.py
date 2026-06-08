"""
Memory Node 评估测试 — 真实 LLM 调用，观察 key 合并质量。

运行方式：
    # 跳过（默认）
    conda run -n agent pytest tests/test_memory_eval.py -v

    # 真实执行（需要 OPENAI_API_KEY）
    conda run -n agent pytest tests/test_memory_eval.py -v -m integration -s
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from biteme.graph.memory import load_memory, memory_node, save_memory
from biteme.config import settings as real_settings

pytestmark = pytest.mark.integration  # 默认跳过，-m integration 才运行


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _make_state(question: str, user_answer: str, llm_reference: str) -> dict:
    return {
        "hitl_flags": ["answerer"],
        "messages": [
            {"speaker": "questioner", "content": question, "retrieved_chunks": []},
            {"speaker": "human", "content": user_answer, "retrieved_chunks": []},
        ],
        "llm_reference_answer": llm_reference,
        "mode": "learn",
        "source_path": "/tmp/fake",
    }


def _run_and_report(state: dict, initial_data: dict, tmp_path: Path) -> dict:
    """写入初始 memory，调用 memory_node（真实 LLM），返回写回后的 memory 内容。"""
    memory_path = tmp_path / "memory.json"
    save_memory(initial_data, memory_path)

    with patch("biteme.graph.memory.settings") as mock_settings:
        mock_settings.biteme_home = tmp_path
        mock_settings.openai_model = real_settings.openai_model
        memory_node(state)

    written = load_memory(memory_path)
    print("\n--- 写回 memory.json ---")
    print(json.dumps(written, ensure_ascii=False, indent=2))
    return written


def _assert_entry_valid(entry: dict) -> None:
    """宽松断言：entry 结构完整，score 在合法范围内。"""
    assert isinstance(entry["aliases"], list)
    assert isinstance(entry["recent_scores"], list)
    assert len(entry["recent_scores"]) >= 1
    assert all(0 <= s <= 10 for s in entry["recent_scores"])
    assert 0.0 <= entry["avg_score"] <= 10.0
    assert isinstance(entry["comments"]["strength"], list)
    assert isinstance(entry["comments"]["weakness"], list)


# ---------------------------------------------------------------------------
# 场景 1 — 应复用已有 key（完全相同概念）
#
# 预置：llm_context_window（上下文窗口 / context window / token 限制）
# 问答：关于上下文窗口长度限制及其影响
# 期望：LLM 复用 llm_context_window，不新建 key
# ---------------------------------------------------------------------------

def test_reuse_existing_key_same_concept(tmp_path):
    initial_data = {
        "entries": {
            "llm_context_window": {
                "aliases": ["上下文窗口", "context window", "token 限制"],
                "recent_scores": [7],
                "avg_score": 7.0,
                "last_update": "2026-01-01",
                "comments": {
                    "strength": ["能说明上下文窗口的基本含义"],
                    "weakness": ["未提及不同模型间的窗口大小差异"],
                },
            }
        }
    }

    state = _make_state(
        question="大语言模型的上下文窗口（context window）是什么？它的长度限制会带来哪些问题？",
        user_answer=(
            "上下文窗口是 LLM 每次推理时能处理的最大 token 数量。"
            "如果输入超过这个限制，模型会截断早期内容，导致遗忘历史对话。"
            "不同模型的窗口大小差异很大，例如 GPT-4 支持 128k tokens。"
        ),
        llm_reference=(
            "上下文窗口（context window）指 LLM 在单次前向传播中能接收的最大 token 数。"
            "超出限制时，通常会截断最早的 token（滑动窗口策略）或报错。"
            "这会导致长对话中早期信息丢失，影响多轮推理质量。"
            "主流模型窗口：GPT-4-turbo 128k、Claude 3 200k、Gemini 1.5 1M。"
        ),
    )

    written = _run_and_report(state, initial_data, tmp_path)

    print("\n[观察] 期望 LLM 复用 llm_context_window，entries 数量应为 1")
    print(f"[结果] entries keys: {list(written['entries'].keys())}")

    assert len(written["entries"]) >= 1
    entry = written["entries"]["llm_context_window"]
    assert len(entry["recent_scores"]) > len(initial_data["entries"]["llm_context_window"]["recent_scores"])
    for entry in written["entries"].values():
        _assert_entry_valid(entry)


# ---------------------------------------------------------------------------
# 场景 2 — 不应复用（话题相关但考察重点不同）
#
# 预置：llm_context_window（上下文窗口 / token 限制）
# 问答：关于 RAG 通过外部知识库弥补知识截止日期
# 期望：LLM 新建 key（如 retrieval_augmented_generation），不复用 llm_context_window
# ---------------------------------------------------------------------------

def test_no_reuse_related_but_different(tmp_path):
    initial_data = {
        "entries": {
            "llm_context_window": {
                "aliases": ["上下文窗口", "context window", "token 限制"],
                "recent_scores": [6],
                "avg_score": 6.0,
                "last_update": "2026-01-01",
                "comments": {"strength": [], "weakness": []},
            }
        }
    }

    state = _make_state(
        question="RAG（检索增强生成）是如何解决大语言模型知识截止日期问题的？",
        user_answer=(
            "RAG 通过在推理时从外部向量数据库检索相关文档片段，"
            "将其拼接到 prompt 中作为上下文，让模型可以利用最新信息回答问题。"
            "这样即使模型的训练数据有截止日期，也能借助外部知识库回答实时问题。"
        ),
        llm_reference=(
            "RAG（Retrieval-Augmented Generation）在推理阶段动态检索外部知识，"
            "将检索结果作为 context 注入 prompt，弥补模型参数化知识的时效性局限。"
            "核心流程：query → embedding → 向量检索 → top-k 文档 → 拼接 prompt → 生成。"
            "与 fine-tuning 不同，RAG 不修改模型权重，知识更新成本低。"
        ),
    )

    written = _run_and_report(state, initial_data, tmp_path)

    print("\n[观察] 期望 LLM 新建 key（不应是 llm_context_window），entries 数量应 >= 2")
    print(f"[结果] entries keys: {list(written['entries'].keys())}")

    assert len(written["entries"]) >= 1
    for entry in written["entries"].values():
        _assert_entry_valid(entry)


# ---------------------------------------------------------------------------
# 场景 3 — 全新 key（memory 为空）
#
# 预置：无任何 key
# 问答：关于 LangGraph checkpointer 的作用和持久化机制
# 期望：LLM 新建有意义的 key（如 langgraph_checkpointer），带合理 aliases
# ---------------------------------------------------------------------------

def test_new_key_empty_memory(tmp_path):
    initial_data = {"entries": {}}

    state = _make_state(
        question="LangGraph 中 checkpointer 的作用是什么？它是如何实现持久化的？",
        user_answer=(
            "checkpointer 用于保存图执行过程中的状态快照，"
            "支持从中断点恢复执行，也就是说可以在 HITL 场景下暂停等待用户输入。"
            "常见实现有 SqliteSaver（本地 SQLite）和 PostgresSaver（云端）。"
        ),
        llm_reference=(
            "LangGraph 的 checkpointer 在每个节点执行后将当前 state 序列化并持久化存储。"
            "当 graph 遇到 interrupt 时，执行挂起；外部输入到达后可从对应 checkpoint 恢复。"
            "这使 Human-in-the-Loop（HITL）场景成为可能，也支持长时运行的 agent 故障恢复。"
            "接口：BaseCheckpointSaver，内置实现包括 SqliteSaver、AsyncSqliteSaver。"
        ),
    )

    written = _run_and_report(state, initial_data, tmp_path)

    print("\n[观察] 期望 LLM 新建至少 1 个有意义的 key，带合理 aliases")
    print(f"[结果] entries keys: {list(written['entries'].keys())}")
    for key, entry in written["entries"].items():
        print(f"  {key}: aliases={entry['aliases']}")

    assert len(written["entries"]) >= 1
    for entry in written["entries"].values():
        _assert_entry_valid(entry)
        assert len(entry["aliases"]) >= 1, "新建 key 应至少有 1 个 alias"


# ---------------------------------------------------------------------------
# 场景 4 — 已有多个 key，测试精准匹配
#
# 预置：llm_context_window / agent_tool_use / rag_pipeline
# 问答：关于 AI Agent 通过 function calling 调用外部 API
# 期望：精准命中 agent_tool_use，不混淆其他两个 key
# ---------------------------------------------------------------------------

def test_precise_match_multiple_keys(tmp_path):
    initial_data = {
        "entries": {
            "llm_context_window": {
                "aliases": ["上下文窗口", "token 限制"],
                "recent_scores": [7],
                "avg_score": 7.0,
                "last_update": "2026-01-01",
                "comments": {"strength": [], "weakness": []},
            },
            "agent_tool_use": {
                "aliases": ["工具调用", "function calling", "tool use"],
                "recent_scores": [6],
                "avg_score": 6.0,
                "last_update": "2026-01-01",
                "comments": {"strength": [], "weakness": []},
            },
            "rag_pipeline": {
                "aliases": ["检索增强生成", "RAG", "向量检索"],
                "recent_scores": [8],
                "avg_score": 8.0,
                "last_update": "2026-01-01",
                "comments": {"strength": [], "weakness": []},
            },
        }
    }

    state = _make_state(
        question="AI Agent 是如何通过 function calling 调用外部 API 的？",
        user_answer=(
            "Agent 会在 prompt 中声明可用的工具和它们的 schema，"
            "模型决定何时调用哪个工具并生成调用参数，"
            "框架负责实际执行工具并将结果返回给模型继续推理。"
        ),
        llm_reference=(
            "Function calling（工具调用）允许 LLM 在推理中输出结构化的函数调用请求，"
            "而非直接生成自然语言答案。框架解析这个请求、执行对应的工具函数、"
            "将返回值作为 tool message 注入对话，模型再继续生成最终回复。"
            "这是构建 ReAct / Agent 的核心机制之一。"
        ),
    )

    written = _run_and_report(state, initial_data, tmp_path)

    print("\n[观察] 期望 LLM 精准复用 agent_tool_use，不应复用 llm_context_window 或 rag_pipeline")
    print(f"[结果] entries keys: {list(written['entries'].keys())}")
    for key, entry in written["entries"].items():
        new_score = entry["recent_scores"][-1] if entry["recent_scores"] else None
        print(f"  {key}: latest_score={new_score}, aliases={entry['aliases']}")

    assert len(written["entries"]) >= 1
    for entry in written["entries"].values():
        _assert_entry_valid(entry)
