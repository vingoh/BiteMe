# Memory Recall & Question Refine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `questioner_node` 内新增 recall + refine 两阶段，使出题策略感知用户历史掌握情况。

**Architecture:** `questioner_node` 保持单一节点。LLM questioner 路径：draft → `recall_memory()` → `refine_question()` → final。HITL questioner 路径：draft（建议问题）→ `recall_memory()` → 展示 Panel → human interrupt。两个新函数都放在 `memory.py`，任何步骤失败均降级到直接使用 draft，不抛异常。

**Tech Stack:** Python 3.10+, LangChain (`ChatOpenAI`, `.with_structured_output()`), Pydantic v2, Rich, pytest, unittest.mock

---

## File Structure

| 文件 | 改动 |
|---|---|
| `biteme/graph/memory.py` | 新增 `RecalledEntry`、`MemoryRecallResult` 模型；新增辅助格式化函数；新增 `recall_memory()`、`refine_question()` |
| `biteme/graph/prompts.py` | 新增 `MEMORY_RECALL_PROMPT`、`MEMORY_REFINE_PROMPT`；`get_prompts()` 新增 `"recall"`、`"refine"` key |
| `biteme/graph/nodes.py` | 修改 `questioner_node()` LLM 路径和 HITL 路径，插入 Phase 2/3 |
| `tests/test_memory_recall.py` | 新建，覆盖 `recall_memory`、`refine_question`、格式化函数的单元测试 |

---

## Task 1: Pydantic 模型 + 格式化辅助函数

**Files:**
- Modify: `biteme/graph/memory.py`
- Create: `tests/test_memory_recall.py`

### 背景

`recall_memory()` 需要把 `MemoryFile` 格式化成字符串传给 LLM，`refine_question()` 需要把 `list[RecalledEntry]` + `MemoryFile` 格式化成带分数/日期/comments 的字符串。这两个格式化函数是纯函数，可以先写并完整测试，不依赖 LLM mock。

- [ ] **Step 1.1: 在 `memory.py` 中新增 Pydantic 模型**

在 `MemoryUpdates` 模型定义之后（约第 60 行），追加：

```python
class RecalledEntry(BaseModel):
    key: str
    relevance_reason: str  # must cite alias or comment text as evidence

class MemoryRecallResult(BaseModel):
    recalled: list[RecalledEntry]  # max 3
```

- [ ] **Step 1.2: 新增格式化辅助函数 `_format_memory_entries_for_recall()`**

在 `_parse_memory_updates()` 之后添加：

```python
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
```

- [ ] **Step 1.3: 新增格式化辅助函数 `_format_recalled_entries_for_refine()`**

紧接着添加：

```python
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
```

- [ ] **Step 1.4: 写测试（新建 `tests/test_memory_recall.py`）**

```python
import pytest
from biteme.graph.memory import (
    RecalledEntry,
    MemoryRecallResult,
    _format_memory_entries_for_recall,
    _format_recalled_entries_for_refine,
)


SAMPLE_MEMORY = {
    "entries": {
        "bpe_tokenization": {
            "aliases": ["BPE", "Byte Pair Encoding", "子词分词"],
            "recent_scores": [3, 4],
            "avg_score": 3.5,
            "last_update": "2026-05-20",
            "comments": {
                "strength": ["用户能说出 BPE 的基本流程"],
                "weakness": ["用户未解释 OOV 处理方式", "未提到 vocabulary size 的影响"],
            },
        },
        "gradient_clipping": {
            "aliases": ["梯度裁剪", "clip_grad_norm"],
            "recent_scores": [8, 9],
            "avg_score": 8.5,
            "last_update": "2026-06-10",
            "comments": {
                "strength": ["正确描述了 clip_grad_norm 的用法"],
                "weakness": [],
            },
        },
    }
}


def test_format_memory_entries_includes_key_and_aliases():
    result = _format_memory_entries_for_recall(SAMPLE_MEMORY)
    assert "bpe_tokenization" in result
    assert "BPE" in result
    assert "Byte Pair Encoding" in result


def test_format_memory_entries_includes_recent_comments_only():
    # weakness 有 2 条，全部应出现（≤3）
    result = _format_memory_entries_for_recall(SAMPLE_MEMORY)
    assert "未解释 OOV 处理方式" in result
    assert "未提到 vocabulary size 的影响" in result


def test_format_memory_entries_empty():
    result = _format_memory_entries_for_recall({"entries": {}})
    assert result == ""


def test_format_recalled_entries_includes_score_and_date():
    recalled = [RecalledEntry(key="bpe_tokenization", relevance_reason='aliases 含 "BPE"')]
    result = _format_recalled_entries_for_refine(recalled, SAMPLE_MEMORY)
    assert "avg_score: 3.5" in result
    assert "last_update: 2026-05-20" in result
    assert 'aliases 含 "BPE"' in result


def test_format_recalled_entries_skips_missing_key():
    recalled = [RecalledEntry(key="nonexistent_key", relevance_reason="test")]
    result = _format_recalled_entries_for_refine(recalled, SAMPLE_MEMORY)
    assert result == ""


def test_format_recalled_entries_includes_weakness():
    recalled = [RecalledEntry(key="bpe_tokenization", relevance_reason="test")]
    result = _format_recalled_entries_for_refine(recalled, SAMPLE_MEMORY)
    assert "未解释 OOV 处理方式" in result
```

- [ ] **Step 1.5: 运行测试，确认全部通过**

```bash
conda run -n agent pytest tests/test_memory_recall.py -v
```

Expected: 6 tests PASS

- [ ] **Step 1.6: Commit**

```bash
git add biteme/graph/memory.py tests/test_memory_recall.py
git commit -m "feat: add RecalledEntry model and formatting helpers for memory recall"
```

---

## Task 2: Prompts

**Files:**
- Modify: `biteme/graph/prompts.py`

- [ ] **Step 2.1: 在 `prompts.py` 中追加两个 prompt 常量**

在 `MEMORY_UPDATER` 之后，`MEMORY_PROMPT_VARIANTS` 之前添加：

```python
MEMORY_RECALL_PROMPT = """\
你是一个记忆检索助手。根据【草稿问题】，从【已有记忆】中选出相关性最高的至多 3 条。

## 草稿问题
{draft_question}

## 已有记忆
{memory_entries}
（每条包含：key、aliases、comments.strength、comments.weakness）

## 判断标准

### 标准一：Alias 直接命中
entry 的 aliases 中有与草稿问题核心概念相同或高度重叠的说法，即便措辞不同。

例子（草稿问题："BPE tokenization 是如何处理 OOV 词的？"）：
- aliases 含 ["BPE", "Byte Pair Encoding", "子词分词"] → ✅ 直接命中
- aliases 含 ["tokenizer", "分词器"] → ⚠️ 需结合 comments 判断，单靠 alias 不够
- aliases 含 ["Transformer 架构", "注意力机制"] → ❌ 不相关

### 标准二：Comments 话题关联
即使 alias 未直接命中，entry 的 strength 或 weakness comments 中提到了草稿问题涉及的
具体概念、机制、或相关术语，也视为相关。

例子（草稿问题："梯度裁剪（gradient clipping）在训练中的作用？"）：
- weakness 含 "用户没有解释梯度爆炸的触发条件" → ✅ 梯度爆炸是梯度裁剪要解决的问题，高度关联
- strength 含 "用户正确描述了 clip_grad_norm 的用法" → ✅ 直接提到相关 API
- comments 含 "用户对反向传播过程描述完整" → ⚠️ 上下游相关但不直接，仅在无更好候选时考虑
- comments 含 "用户对学习率调度理解到位" → ❌ 同属优化领域但话题不同

### 不算相关的情况
- 仅因同属一个大领域（例如都是"深度学习"、"Python"、"NLP"）→ ❌
- key 名称中有相似词但 aliases 和 comments 均无交集 → ❌
- 相关性纯靠猜测/推断，没有 alias 或 comment 中的直接证据 → ❌

## 输出
只输出 JSON，不含 markdown 围栏：
{{"recalled": [{{"key": "...", "relevance_reason": "..."}}, ...]}}
relevance_reason 必须引用具体 alias 或 comment 中的文字作为依据。
最多 3 条，完全不相关时输出 {{"recalled": []}}。
"""

MEMORY_REFINE_PROMPT = """\
你是一位提问优化助手。根据用户的历史掌握情况，对【草稿问题】进行调整，
使问题更有针对性地帮助用户查漏补缺。

## 草稿问题
{draft_question}

## 相关历史记忆（按相关性排序）
{recalled_entries}
（每条包含：key、avg_score(0-10)、last_update、relevance_reason、
  comments.strength（用户在此话题上的具体优点）、
  comments.weakness（用户在此话题上的具体错误或遗漏））

## 调整规则

### 基于分数与日期
- avg_score ≥ 7 且 last_update 在 14 天内：用户已近期掌握，转向相邻话题或追问更深层细节
- avg_score ≤ 4：薄弱环节，保持问题方向，可适当简化难度从基础考起
- last_update 超过 14 天（无论分数）：久未复习，保持或强化原方向
- 若所有召回记忆均为高分且近期：在同领域内换一个新角度提问

### 基于 Comments（优先级高于分数规则）
- weakness 中有具体错误或遗漏 → 针对该错误/遗漏设计问题，让用户有机会弥补
  例：weakness "没有解释梯度爆炸的触发条件" → 问题可聚焦在梯度爆炸的触发机制上
- strength 中有某项能力已熟练掌握 → 避免重复考察该项，转向 weakness 或更深层问题
  例：strength "正确描述了 clip_grad_norm 的用法" → 不再考用法，改考原理或边界情况
- weakness 和 strength 均为空 → 退回分数/日期规则

## 输出
只输出最终问题本身，不含任何前缀或解释。
"""
```

- [ ] **Step 2.2: 更新 `get_prompts()` 加入 `"recall"` 和 `"refine"` key**

修改 `get_prompts()` 函数（两个分支都加）：

```python
def get_prompts(mode: str) -> dict[str, str]:
    if mode == "learn":
        return {
            "questioner": LEARN_QUESTIONER,
            "answerer": LEARN_ANSWERER,
            "planner": LEARN_PLANNER,
            "memory": get_memory_prompt("default"),
            "recall": MEMORY_RECALL_PROMPT,
            "refine": MEMORY_REFINE_PROMPT,
        }
    return {
        "questioner": INTERVIEW_QUESTIONER,
        "answerer": INTERVIEW_ANSWERER,
        "planner": INTERVIEW_PLANNER,
        "memory": get_memory_prompt("default"),
        "recall": MEMORY_RECALL_PROMPT,
        "refine": MEMORY_REFINE_PROMPT,
    }
```

- [ ] **Step 2.3: 验证 prompts 包含正确的 format 占位符**

```bash
conda run -n agent python -c "
from biteme.graph.prompts import get_prompts
p = get_prompts('interview')
assert '{draft_question}' in p['recall']
assert '{memory_entries}' in p['recall']
assert '{draft_question}' in p['refine']
assert '{recalled_entries}' in p['refine']
assert 'recall' in p and 'refine' in p
print('OK')
"
```

Expected output: `OK`

- [ ] **Step 2.4: Commit**

```bash
git add biteme/graph/prompts.py
git commit -m "feat: add MEMORY_RECALL_PROMPT and MEMORY_REFINE_PROMPT"
```

---

## Task 3: `recall_memory()` 函数

**Files:**
- Modify: `biteme/graph/memory.py`
- Modify: `tests/test_memory_recall.py`

- [ ] **Step 3.1: 写失败测试**

在 `tests/test_memory_recall.py` 末尾追加：

```python
from unittest.mock import patch, MagicMock
from biteme.graph.memory import recall_memory


def test_recall_memory_empty_memory_returns_empty():
    result = recall_memory("BPE tokenization 是如何处理 OOV 词的？", {"entries": {}})
    assert result == []


def test_recall_memory_empty_draft_returns_empty():
    result = recall_memory("", SAMPLE_MEMORY)
    assert result == []


def test_recall_memory_returns_filtered_entries():
    mock_result = MemoryRecallResult(recalled=[
        RecalledEntry(key="bpe_tokenization", relevance_reason='aliases 含 "BPE"'),
        RecalledEntry(key="nonexistent_key", relevance_reason="test"),  # should be filtered
    ])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_result

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.with_structured_output.return_value = mock_llm
        result = recall_memory("BPE tokenization 是如何处理 OOV 词的？", SAMPLE_MEMORY)

    assert len(result) == 1
    assert result[0].key == "bpe_tokenization"


def test_recall_memory_llm_failure_returns_empty():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM error")

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.with_structured_output.return_value = mock_llm
        result = recall_memory("some question", SAMPLE_MEMORY)

    assert result == []


def test_recall_memory_returns_at_most_3():
    # LLM 返回 3 条都在 memory 中
    three_entries = [
        RecalledEntry(key="bpe_tokenization", relevance_reason="test"),
        RecalledEntry(key="gradient_clipping", relevance_reason="test"),
        RecalledEntry(key="bpe_tokenization", relevance_reason="duplicate"),  # key重复但还是过滤后≤3
    ]
    mock_result = MemoryRecallResult(recalled=three_entries)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_result

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.with_structured_output.return_value = mock_llm
        result = recall_memory("some question", SAMPLE_MEMORY)

    assert len(result) <= 3
```

- [ ] **Step 3.2: 运行测试，确认失败（函数尚未实现）**

```bash
conda run -n agent pytest tests/test_memory_recall.py::test_recall_memory_empty_memory_returns_empty -v
```

Expected: FAIL with `ImportError` or `AttributeError`

- [ ] **Step 3.3: 实现 `recall_memory()`**

在 `memory.py` 的 `invoke_memory_update()` 之前添加：

```python
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
    from .prompts import get_prompts  # local import to avoid circular at module level
    # use a neutral mode for recall (same prompt regardless of mode)
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
```

注意：`MEMORY_RECALL_PROMPT` 直接在模块顶层从 `prompts.py` 导入会产生循环导入（`prompts.py` 不导入 `memory.py`，所以实际上没有循环）。确认后改为模块级导入：在 `memory.py` 顶部 `from .prompts import get_prompts` 改为只在函数内部导入，或直接在文件顶部加：

```python
from .prompts import MEMORY_RECALL_PROMPT, MEMORY_REFINE_PROMPT
```

检查一下现有 imports：`memory.py` 目前已经 `from .prompts import get_prompts`，所以直接加到该行即可：

```python
from .prompts import get_prompts, MEMORY_RECALL_PROMPT, MEMORY_REFINE_PROMPT
```

（这行在 `recall_memory()` 函数体中不需要局部 import，删掉函数内的局部 import 注释。）

- [ ] **Step 3.4: 运行所有 recall_memory 测试**

```bash
conda run -n agent pytest tests/test_memory_recall.py -k "recall_memory" -v
```

Expected: 5 tests PASS

- [ ] **Step 3.5: Commit**

```bash
git add biteme/graph/memory.py tests/test_memory_recall.py
git commit -m "feat: implement recall_memory()"
```

---

## Task 4: `refine_question()` 函数

**Files:**
- Modify: `biteme/graph/memory.py`
- Modify: `tests/test_memory_recall.py`

- [ ] **Step 4.1: 写失败测试**

在 `tests/test_memory_recall.py` 末尾追加：

```python
from biteme.graph.memory import refine_question


SAMPLE_RECALLED = [
    RecalledEntry(key="bpe_tokenization", relevance_reason='aliases 含 "BPE"'),
]


def test_refine_question_returns_llm_output():
    mock_response = MagicMock()
    mock_response.content = "BPE 在遇到 OOV 词时会如何进行子词切分？请从 vocabulary 构建过程解释。"

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.invoke.return_value = mock_response
        result = refine_question("BPE 如何工作？", SAMPLE_RECALLED, SAMPLE_MEMORY)

    assert result == "BPE 在遇到 OOV 词时会如何进行子词切分？请从 vocabulary 构建过程解释。"


def test_refine_question_returns_draft_on_llm_failure():
    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.invoke.side_effect = Exception("LLM error")
        result = refine_question("BPE 如何工作？", SAMPLE_RECALLED, SAMPLE_MEMORY)

    assert result == "BPE 如何工作？"


def test_refine_question_returns_draft_on_empty_response():
    mock_response = MagicMock()
    mock_response.content = "   "  # blank

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.invoke.return_value = mock_response
        result = refine_question("BPE 如何工作？", SAMPLE_RECALLED, SAMPLE_MEMORY)

    assert result == "BPE 如何工作？"


def test_refine_question_uses_comments_in_prompt():
    """Verify the prompt sent to LLM includes comments content."""
    mock_response = MagicMock()
    mock_response.content = "refined question"
    captured_prompt = []

    def capture_invoke(messages):
        captured_prompt.append(messages[0].content)
        return mock_response

    with patch("biteme.graph.memory.ChatOpenAI") as mock_chat:
        mock_chat.return_value.invoke.side_effect = capture_invoke
        refine_question("BPE 如何工作？", SAMPLE_RECALLED, SAMPLE_MEMORY)

    assert len(captured_prompt) == 1
    assert "未解释 OOV 处理方式" in captured_prompt[0]  # weakness 出现在 prompt 中
    assert "用户能说出 BPE 的基本流程" in captured_prompt[0]  # strength 出现在 prompt 中
```

- [ ] **Step 4.2: 运行测试，确认失败**

```bash
conda run -n agent pytest tests/test_memory_recall.py::test_refine_question_returns_llm_output -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 4.3: 实现 `refine_question()`**

在 `recall_memory()` 之后添加：

```python
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
```

- [ ] **Step 4.4: 运行所有 refine_question 测试**

```bash
conda run -n agent pytest tests/test_memory_recall.py -k "refine_question" -v
```

Expected: 4 tests PASS

- [ ] **Step 4.5: 运行所有 memory recall 测试**

```bash
conda run -n agent pytest tests/test_memory_recall.py -v
```

Expected: 全部 PASS（共约 15 个测试）

- [ ] **Step 4.6: Commit**

```bash
git add biteme/graph/memory.py tests/test_memory_recall.py
git commit -m "feat: implement refine_question()"
```

---

## Task 5: 修改 `questioner_node()` — LLM 路径

**Files:**
- Modify: `biteme/graph/nodes.py`
- Modify: `tests/test_memory_recall.py`

在 LLM questioner 路径（`else` 分支，第 82-129 行）中，`stream_agent(...)` 产生 `question_text` 之后，插入 recall + refine。

- [ ] **Step 5.1: 写失败测试**

在 `tests/test_memory_recall.py` 末尾追加：

```python
from unittest.mock import patch, MagicMock
from biteme.graph.nodes import questioner_node
from biteme.graph.state import SessionState


def _make_state(**overrides) -> SessionState:
    base: SessionState = {
        "mode": "interview",
        "messages": [],
        "current_speaker": "questioner",
        "hitl_flags": [],
        "turn_count": 0,
        "max_turns": 5,
        "context_strategy": "direct",
        "source_path": "/tmp/fake",
        "outline": ["第一个话题"],
        "llm_reference_answer": "",
    }
    base.update(overrides)
    return base


def test_questioner_node_llm_path_calls_recall_and_refine(tmp_path):
    """When memory has entries, questioner should call recall_memory and refine_question."""
    state = _make_state()
    draft = "BPE 如何处理 OOV？"
    refined = "BPE 在遇到未知词时的子词切分机制是什么？"
    recalled = [RecalledEntry(key="bpe_tokenization", relevance_reason="test")]

    with (
        patch("biteme.graph.nodes.stream_agent", return_value=draft),
        patch("biteme.graph.nodes.load_memory", return_value=SAMPLE_MEMORY),
        patch("biteme.graph.nodes.recall_memory", return_value=recalled) as mock_recall,
        patch("biteme.graph.nodes.refine_question", return_value=refined) as mock_refine,
        patch("biteme.graph.nodes.create_provider") as mock_provider,
        patch("biteme.graph.nodes.create_agent"),
    ):
        mock_provider.return_value.get_overview.return_value = ["context"]
        result = questioner_node(state)

    mock_recall.assert_called_once_with(draft, SAMPLE_MEMORY)
    mock_refine.assert_called_once_with(draft, recalled, SAMPLE_MEMORY)
    assert result["messages"][-1]["content"] == refined


def test_questioner_node_llm_path_uses_draft_when_no_recall(tmp_path):
    """When recall returns empty, questioner uses draft directly."""
    state = _make_state()
    draft = "BPE 如何处理 OOV？"

    with (
        patch("biteme.graph.nodes.stream_agent", return_value=draft),
        patch("biteme.graph.nodes.load_memory", return_value={"entries": {}}),
        patch("biteme.graph.nodes.recall_memory", return_value=[]) as mock_recall,
        patch("biteme.graph.nodes.refine_question") as mock_refine,
        patch("biteme.graph.nodes.create_provider") as mock_provider,
        patch("biteme.graph.nodes.create_agent"),
    ):
        mock_provider.return_value.get_overview.return_value = ["context"]
        result = questioner_node(state)

    mock_recall.assert_called_once()
    mock_refine.assert_not_called()
    assert result["messages"][-1]["content"] == draft
```

- [ ] **Step 5.2: 运行测试，确认失败**

```bash
conda run -n agent pytest tests/test_memory_recall.py::test_questioner_node_llm_path_calls_recall_and_refine -v
```

Expected: FAIL（`questioner_node` 尚未调用 `recall_memory`）

- [ ] **Step 5.3: 修改 `nodes.py`，在顶部导入新函数**

在 `nodes.py` 的 import 区域，`from .memory import memory_node` 改为：

```python
from .memory import memory_node, recall_memory, refine_question, load_memory
```

- [ ] **Step 5.4: 修改 `questioner_node()` LLM 路径**

找到 `questioner_node()` 的 `else` 分支（约第 82 行起）。在 `stream_agent(...)` 调用之后，`turn = {"speaker": ...}` 之前插入：

```python
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

        # Phase 2: memory recall
        memory_path = settings.biteme_home / "memory.json"
        memory_data = load_memory(memory_path)
        recalled = recall_memory(question_text, memory_data)

        # Phase 3: refine based on recalled memories
        if recalled:
            question_text = refine_question(question_text, recalled, memory_data)

        turn = {"speaker": "questioner", "content": question_text, "retrieved_chunks": []}
```

- [ ] **Step 5.5: 运行 LLM 路径测试**

```bash
conda run -n agent pytest tests/test_memory_recall.py -k "questioner_node_llm" -v
```

Expected: 2 tests PASS

- [ ] **Step 5.6: 运行完整 test suite 确认无回归**

```bash
conda run -n agent pytest tests/test_memory_recall.py tests/test_memory.py tests/test_graph_nodes.py -v
```

Expected: 全部 PASS

- [ ] **Step 5.7: Commit**

```bash
git add biteme/graph/nodes.py tests/test_memory_recall.py
git commit -m "feat: integrate recall+refine into questioner_node LLM path"
```

---

## Task 6: 修改 `questioner_node()` — HITL 路径

**Files:**
- Modify: `biteme/graph/nodes.py`
- Modify: `tests/test_memory_recall.py`

HITL 路径不做 refine（人类自行决策），只在 `interrupt()` 前展示召回的记忆 Panel。

- [ ] **Step 6.1: 写失败测试**

在 `tests/test_memory_recall.py` 末尾追加：

```python
def test_questioner_node_hitl_shows_recall_panel():
    """HITL path: when memory has entries, a panel with recalled memories is printed."""
    state = _make_state(hitl_flags=["questioner"])
    recalled = [RecalledEntry(key="bpe_tokenization", relevance_reason='aliases 含 "BPE"')]

    printed_panels = []

    class FakeConsole:
        def print(self, renderable, **kwargs):
            printed_panels.append(renderable)

    with (
        patch("biteme.graph.nodes.load_memory", return_value=SAMPLE_MEMORY),
        patch("biteme.graph.nodes.recall_memory", return_value=recalled),
        patch("biteme.graph.nodes.Console", return_value=FakeConsole()),
        patch("biteme.graph.nodes.interrupt", return_value="my question"),
    ):
        result = questioner_node(state)

    assert result["messages"][-1]["content"] == "my question"
    panel_texts = [str(p) for p in printed_panels]
    assert any("bpe_tokenization" in t for t in panel_texts)


def test_questioner_node_hitl_silent_when_no_recall():
    """HITL path: when recalled is empty, no memory panel is shown."""
    state = _make_state(hitl_flags=["questioner"])
    printed_panels = []

    class FakeConsole:
        def print(self, renderable, **kwargs):
            printed_panels.append(str(renderable))

    with (
        patch("biteme.graph.nodes.load_memory", return_value={"entries": {}}),
        patch("biteme.graph.nodes.recall_memory", return_value=[]),
        patch("biteme.graph.nodes.Console", return_value=FakeConsole()),
        patch("biteme.graph.nodes.interrupt", return_value="my question"),
    ):
        result = questioner_node(state)

    assert not any("相关记忆参考" in t for t in printed_panels)
```

- [ ] **Step 6.2: 运行测试，确认失败**

```bash
conda run -n agent pytest tests/test_memory_recall.py::test_questioner_node_hitl_shows_recall_panel -v
```

Expected: FAIL

- [ ] **Step 6.3: 修改 `questioner_node()` HITL 路径**

找到 HITL 路径（`if "questioner" in state["hitl_flags"]:` 分支，约第 67 行）。在 `human_text = interrupt(prompt_msg)` **之前**插入 recall + Panel 展示：

```python
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
            draft_for_recall = suggested
        else:
            prompt_msg = "请输入你的问题："
            draft_for_recall = ""

        # Phase 2: memory recall (only if we have a draft to compare against)
        if draft_for_recall:
            memory_path = settings.biteme_home / "memory.json"
            memory_data = load_memory(memory_path)
            recalled = recall_memory(draft_for_recall, memory_data)
            if recalled:
                recall_lines = "\n".join(
                    f"  • {e.key}"
                    f"  avg_score={memory_data['entries'][e.key]['avg_score']}"
                    f"  最近: {memory_data['entries'][e.key]['last_update']}\n"
                    f"    相关依据: {e.relevance_reason}"
                    for e in recalled
                    if e.key in memory_data["entries"]
                )
                Console().print(Panel(recall_lines, title="[magenta]相关记忆参考[/magenta]"))

        human_text = interrupt(prompt_msg)
        turn: Turn = {"speaker": "human", "content": human_text, "retrieved_chunks": []}
```

- [ ] **Step 6.4: 运行 HITL 路径测试**

```bash
conda run -n agent pytest tests/test_memory_recall.py -k "hitl" -v
```

Expected: 2 tests PASS

- [ ] **Step 6.5: 运行全部 memory recall 测试**

```bash
conda run -n agent pytest tests/test_memory_recall.py -v
```

Expected: 全部 PASS（约 21 个测试）

- [ ] **Step 6.6: 运行完整 test suite**

```bash
conda run -n agent pytest tests/ -v --ignore=tests/test_memory_benchmark.py
```

Expected: 全部 PASS（跳过需要真实 LLM 的 integration benchmark）

- [ ] **Step 6.7: Commit**

```bash
git add biteme/graph/nodes.py tests/test_memory_recall.py
git commit -m "feat: show recalled memory panel in HITL questioner path"
```

---

## 完成标志

所有单元测试通过：

```bash
conda run -n agent pytest tests/ -v --ignore=tests/test_memory_benchmark.py
```

运行集成 benchmark 观察 recall/refine 实际效果（可选，需 API key）：

```bash
conda run -n agent pytest tests/test_memory_benchmark.py -m integration -s
```
