# BiteMe 双 Agent 问答系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 构建一个 CLI 驱动的双 Agent 问答系统，支持仓库导读与模拟面试两种模式，具备 HITL 和 RAG/直读双策略。

**架构：** LangGraph 管理两节点对话图（questioner / answerer），ContextProvider 抽象屏蔽 RAG 与直读差异，LanceDB 存索引，SQLite Checkpoint 做会话持久化，Typer + Rich 做 CLI 壳。

**技术栈：** Python 3.11+, LangGraph, LangChain, langchain-openai, LanceDB, Typer, Rich, pytest

**运行环境：** Anaconda，环境名 `agent`。所有安装和运行命令均在该环境下执行。每次开启新终端先激活：

```bash
conda activate agent
```

---

## 文件清单

| 文件 | 职责 |
|------|------|
| `pyproject.toml` | 依赖与入口点声明 |
| `biteme/config.py` | 读取 `.env`，暴露 `settings` 对象 |
| `biteme/context/base.py` | `ContextProvider` 抽象基类 |
| `biteme/context/direct.py` | `DirectProvider`：读文件全文 |
| `biteme/context/rag.py` | `RAGProvider`：LanceDB 检索 |
| `biteme/context/factory.py` | `create_provider()`：auto/direct/rag 策略选择 |
| `biteme/indexing/pipeline.py` | `build_index()`：loader → splitter → embed → LanceDB |
| `biteme/graph/state.py` | `Turn`、`SessionState` TypedDict |
| `biteme/graph/prompts.py` | 四个 system prompt 模板（learn/interview × questioner/answerer） |
| `biteme/graph/nodes.py` | `questioner_node()`、`answerer_node()` |
| `biteme/graph/graph.py` | `build_graph()`：组装 LangGraph StateGraph |
| `biteme/session/manager.py` | `create_session()`、`list_sessions()`、`get_checkpoint_saver()` |
| `biteme/cli.py` | Typer app，`index` / `run` / `resume` / `list` 子命令 |
| `tests/test_context_direct.py` | DirectProvider 单元测试 |
| `tests/test_context_rag.py` | RAGProvider 单元测试（需要临时 LanceDB） |
| `tests/test_context_factory.py` | factory auto/direct/rag 路由测试 |
| `tests/test_graph_nodes.py` | questioner_node / answerer_node 单元测试（mock LLM） |
| `tests/test_session_manager.py` | session CRUD 测试 |
| `tests/test_cli.py` | CLI 命令集成测试（typer.testing.CliRunner） |

---

## Task 1：项目初始化

**文件：**
- 创建：`pyproject.toml`
- 创建：`biteme/__init__.py`、`biteme/context/__init__.py`、`biteme/graph/__init__.py`、`biteme/session/__init__.py`、`biteme/indexing/__init__.py`
- 创建：`tests/__init__.py`
- 创建：`.env.example`

- [ ] **步骤 1：创建 `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "biteme"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2",
    "langchain>=0.3",
    "langchain-openai>=0.2",
    "langchain-community>=0.3",
    "lancedb>=0.13",
    "typer>=0.12",
    "rich>=13",
    "python-dotenv>=1.0",
    "tiktoken>=0.7",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-mock>=3.14"]

[project.scripts]
biteme = "biteme.cli:app"
```

- [ ] **步骤 2：在 `agent` 环境中安装依赖**

```bash
conda activate agent
pip install -e ".[dev]"
```

预期输出：`Successfully installed biteme-0.1.0 ...`（无报错）

- [ ] **步骤 3：创建所有 `__init__.py` 和 `.env.example`**

所有 `__init__.py` 内容为空。`.env.example` 内容：

```
OPENAI_API_KEY=sk-...
BITEME_HOME=~/.biteme
```

- [ ] **步骤 4：提交**

```bash
git add pyproject.toml biteme/ tests/ .env.example
git commit -m "chore: 初始化项目结构与依赖"
```

---

## Task 2：配置模块

**文件：**
- 创建：`biteme/config.py`

- [ ] **步骤 1：写失败测试**

```python
# tests/test_config.py
import os
from biteme.config import settings

def test_biteme_home_default(tmp_path, monkeypatch):
    monkeypatch.delenv("BITEME_HOME", raising=False)
    # 重新导入以触发默认值逻辑
    import importlib, biteme.config as cfg
    importlib.reload(cfg)
    assert str(cfg.settings.biteme_home).endswith(".biteme")

def test_biteme_home_custom(tmp_path, monkeypatch):
    monkeypatch.setenv("BITEME_HOME", str(tmp_path))
    import importlib, biteme.config as cfg
    importlib.reload(cfg)
    assert cfg.settings.biteme_home == tmp_path
```

- [ ] **步骤 2：运行确认失败**

```bash
pytest tests/test_config.py -v
```

预期：`ModuleNotFoundError: No module named 'biteme.config'`

- [ ] **步骤 3：实现 `biteme/config.py`**

```python
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    biteme_home: Path = Path(os.getenv("BITEME_HOME", "~/.biteme")).expanduser()
    indexes_dir: Path = property(lambda self: self.biteme_home / "indexes")
    sessions_dir: Path = property(lambda self: self.biteme_home / "sessions")

    def ensure_dirs(self) -> None:
        self.biteme_home.mkdir(parents=True, exist_ok=True)
        (self.biteme_home / "indexes").mkdir(exist_ok=True)
        (self.biteme_home / "sessions").mkdir(exist_ok=True)

settings = Settings()
```

- [ ] **步骤 4：运行确认通过**

```bash
pytest tests/test_config.py -v
```

预期：`2 passed`

- [ ] **步骤 5：提交**

```bash
git add biteme/config.py tests/test_config.py
git commit -m "feat: 添加配置模块"
```

---

## Task 3：ContextProvider 抽象基类与 DirectProvider

**文件：**
- 创建：`biteme/context/base.py`
- 创建：`biteme/context/direct.py`
- 创建：`tests/test_context_direct.py`

- [ ] **步骤 1：写失败测试**

```python
# tests/test_context_direct.py
import pytest
from pathlib import Path
from biteme.context.direct import DirectProvider

def test_retrieve_returns_full_content(tmp_path):
    f = tmp_path / "hello.py"
    f.write_text("def hello(): return 'world'")
    provider = DirectProvider(source_path=str(tmp_path))
    chunks = provider.retrieve("hello function")
    assert len(chunks) == 1
    assert "def hello" in chunks[0]

def test_retrieve_multiple_files(tmp_path):
    (tmp_path / "a.py").write_text("class A: pass")
    (tmp_path / "b.md").write_text("# Doc")
    provider = DirectProvider(source_path=str(tmp_path))
    chunks = provider.retrieve("anything")
    assert len(chunks) == 2

def test_retrieve_single_file(tmp_path):
    f = tmp_path / "resume.md"
    f.write_text("# 张三\n## 经历")
    provider = DirectProvider(source_path=str(f))
    chunks = provider.retrieve("经历")
    assert "张三" in chunks[0]

def test_get_overview_returns_content(tmp_path):
    (tmp_path / "main.py").write_text("def main(): pass")
    (tmp_path / "README.md").write_text("# Project")
    provider = DirectProvider(source_path=str(tmp_path))
    chunks = provider.get_overview()
    assert len(chunks) == 2
    contents = "\n".join(chunks)
    assert "def main" in contents
```

- [ ] **步骤 2：运行确认失败**

```bash
pytest tests/test_context_direct.py -v
```

预期：`ImportError` 或 `ModuleNotFoundError`

- [ ] **步骤 3：实现 `biteme/context/base.py`**

```python
from abc import ABC, abstractmethod

class ContextProvider(ABC):
    @abstractmethod
    def get_overview(self) -> list[str]:
        """冷启动用：不需要 query，直接返回源内容概览片段。"""
        ...

    @abstractmethod
    def retrieve(self, query: str) -> list[str]:
        """根据 query 返回相关内容片段列表。"""
        ...
```

- [ ] **步骤 4：实现 `biteme/context/direct.py`**

```python
from pathlib import Path
from .base import ContextProvider

_SUPPORTED_EXTENSIONS = {".py", ".md", ".txt", ".ts", ".js", ".go", ".rs", ".java", ".yaml", ".toml", ".json"}

class DirectProvider(ContextProvider):
    def __init__(self, source_path: str) -> None:
        self._path = Path(source_path)

    def _read_all(self) -> list[str]:
        if self._path.is_file():
            return [self._path.read_text(errors="ignore")]
        chunks = []
        for f in sorted(self._path.rglob("*")):
            if f.is_file() and f.suffix in _SUPPORTED_EXTENSIONS:
                chunks.append(f.read_text(errors="ignore"))
        return chunks

    def get_overview(self) -> list[str]:
        return self._read_all()

    def retrieve(self, query: str) -> list[str]:
        # DirectProvider 内容全量在内存里，query 不影响结果
        return self._read_all()
```

- [ ] **步骤 5：运行确认通过**

```bash
pytest tests/test_context_direct.py -v
```

预期：`4 passed`

- [ ] **步骤 6：提交**

```bash
git add biteme/context/base.py biteme/context/direct.py tests/test_context_direct.py
git commit -m "feat: 添加 ContextProvider 基类与 DirectProvider"
```

---

## Task 4：建索引管线（IndexingPipeline）

**文件：**
- 创建：`biteme/indexing/pipeline.py`
- 创建：`tests/test_indexing.py`

- [ ] **步骤 1：写失败测试**

```python
# tests/test_indexing.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from biteme.indexing.pipeline import build_index, estimate_tokens

def test_estimate_tokens_small_file(tmp_path):
    f = tmp_path / "small.py"
    f.write_text("def foo(): pass")
    count = estimate_tokens(str(tmp_path))
    assert count > 0
    assert count < 1000

def test_build_index_creates_lancedb(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def main(): print('hello')")
    db_path = tmp_path / "db"

    # mock OpenAI embeddings 避免真实 API 调用
    mock_embed = MagicMock()
    mock_embed.embed_documents.return_value = [[0.1] * 1536]
    with patch("biteme.indexing.pipeline.OpenAIEmbeddings", return_value=mock_embed):
        build_index(source_path=str(src), db_path=str(db_path))

    assert db_path.exists()
```

- [ ] **步骤 2：运行确认失败**

```bash
pytest tests/test_indexing.py -v
```

预期：`ImportError`

- [ ] **步骤 3：实现 `biteme/indexing/pipeline.py`**

```python
from pathlib import Path
import tiktoken
import lancedb
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

_SUPPORTED_GLOB = "**/*.{py,md,txt,ts,js,go,rs,java,yaml,toml,json}"

def estimate_tokens(source_path: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    path = Path(source_path)
    files = [path] if path.is_file() else list(path.rglob("*"))
    for f in files:
        if f.is_file() and f.suffix in {".py", ".md", ".txt", ".ts", ".js", ".go", ".rs", ".java", ".yaml", ".toml", ".json"}:
            try:
                total += len(enc.encode(f.read_text(errors="ignore")))
            except Exception:
                pass
    return total

def build_index(source_path: str, db_path: str) -> None:
    loader = DirectoryLoader(
        source_path,
        glob=_SUPPORTED_GLOB,
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        silent_errors=True,
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vectors = embeddings.embed_documents(texts)

    db = lancedb.connect(db_path)
    data = [
        {"vector": v, "text": t, "source": m.get("source", "")}
        for v, t, m in zip(vectors, texts, metadatas)
    ]
    db.create_table("chunks", data=data, mode="overwrite")
```

- [ ] **步骤 4：运行确认通过**

```bash
pytest tests/test_indexing.py -v
```

预期：`2 passed`

- [ ] **步骤 5：提交**

```bash
git add biteme/indexing/pipeline.py tests/test_indexing.py
git commit -m "feat: 添加建索引管线（LanceDB + OpenAI embeddings）"
```

---

## Task 5：RAGProvider

**文件：**
- 创建：`biteme/context/rag.py`
- 创建：`tests/test_context_rag.py`

- [ ] **步骤 1：写失败测试**

```python
# tests/test_context_rag.py
import pytest
from unittest.mock import patch, MagicMock
from biteme.context.rag import RAGProvider

def test_retrieve_returns_chunks(tmp_path):
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 1536

    mock_table = MagicMock()
    mock_table.search.return_value.limit.return_value.to_list.return_value = [
        {"text": "def foo(): pass", "score": 0.9},
        {"text": "def bar(): pass", "score": 0.8},
    ]
    mock_db = MagicMock()
    mock_db.open_table.return_value = mock_table

    with patch("biteme.context.rag.OpenAIEmbeddings", return_value=mock_embed), \
         patch("biteme.context.rag.lancedb.connect", return_value=mock_db):
        provider = RAGProvider(db_path=str(tmp_path / "db"), top_k=2)
        chunks = provider.retrieve("find foo")

    assert len(chunks) == 2
    assert "def foo" in chunks[0]

def test_get_overview_does_not_use_vector_search(tmp_path):
    mock_embed = MagicMock()
    mock_table = MagicMock()
    mock_table.to_pandas.return_value.__getitem__.return_value.tolist.return_value = [
        "def foo(): pass",
        "def bar(): pass",
        "class Baz: pass",
    ]
    mock_db = MagicMock()
    mock_db.open_table.return_value = mock_table

    with patch("biteme.context.rag.OpenAIEmbeddings", return_value=mock_embed), \
         patch("biteme.context.rag.lancedb.connect", return_value=mock_db):
        provider = RAGProvider(db_path=str(tmp_path / "db"), top_k=5)
        chunks = provider.get_overview()

    mock_embed.embed_query.assert_not_called()   # 不做向量搜索
    assert len(chunks) > 0
```

- [ ] **步骤 2：运行确认失败**

```bash
pytest tests/test_context_rag.py -v
```

预期：`ImportError`

- [ ] **步骤 3：实现 `biteme/context/rag.py`**

```python
import lancedb
from langchain_openai import OpenAIEmbeddings
from .base import ContextProvider

_OVERVIEW_ROWS = 10  # get_overview 直接取前 N 行，不做向量搜索

class RAGProvider(ContextProvider):
    def __init__(self, db_path: str, top_k: int = 5) -> None:
        self._db_path = db_path
        self._top_k = top_k
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._db = lancedb.connect(db_path)
        self._table = self._db.open_table("chunks")

    def get_overview(self) -> list[str]:
        # 直接扫表取前 N 条，不做向量搜索，避免无意义 query 污染结果
        rows = self._table.to_pandas()["text"].tolist()
        return rows[:_OVERVIEW_ROWS]

    def retrieve(self, query: str) -> list[str]:
        vector = self._embeddings.embed_query(query)
        results = self._table.search(vector).limit(self._top_k).to_list()
        return [r["text"] for r in results]
```

- [ ] **步骤 4：运行确认通过**

```bash
pytest tests/test_context_rag.py -v
```

预期：`2 passed`

- [ ] **步骤 5：提交**

```bash
git add biteme/context/rag.py tests/test_context_rag.py
git commit -m "feat: 添加 RAGProvider（LanceDB 检索）"
```

---

## Task 6：ContextProvider Factory

**文件：**
- 创建：`biteme/context/factory.py`
- 创建：`tests/test_context_factory.py`

- [ ] **步骤 1：写失败测试**

```python
# tests/test_context_factory.py
import pytest
from unittest.mock import patch
from biteme.context.factory import create_provider
from biteme.context.direct import DirectProvider
from biteme.context.rag import RAGProvider

def test_strategy_direct_returns_direct_provider(tmp_path):
    (tmp_path / "f.py").write_text("x = 1")
    provider = create_provider(source_path=str(tmp_path), strategy="direct", db_path=str(tmp_path / "db"))
    assert isinstance(provider, DirectProvider)

def test_strategy_rag_returns_rag_provider(tmp_path):
    from unittest.mock import MagicMock
    with patch("biteme.context.factory.RAGProvider") as mock_rag:
        mock_rag.return_value = MagicMock(spec=RAGProvider)
        provider = create_provider(source_path=str(tmp_path), strategy="rag", db_path=str(tmp_path / "db"))
        mock_rag.assert_called_once()

def test_strategy_auto_small_uses_direct(tmp_path):
    (tmp_path / "small.py").write_text("x = 1")
    with patch("biteme.context.factory.estimate_tokens", return_value=1000):
        provider = create_provider(source_path=str(tmp_path), strategy="auto", db_path=str(tmp_path / "db"))
    assert isinstance(provider, DirectProvider)

def test_strategy_auto_large_uses_rag(tmp_path):
    from unittest.mock import MagicMock
    with patch("biteme.context.factory.estimate_tokens", return_value=200_000), \
         patch("biteme.context.factory.RAGProvider") as mock_rag:
        mock_rag.return_value = MagicMock(spec=RAGProvider)
        create_provider(source_path=str(tmp_path), strategy="auto", db_path=str(tmp_path / "db"))
        mock_rag.assert_called_once()
```

- [ ] **步骤 2：运行确认失败**

```bash
pytest tests/test_context_factory.py -v
```

预期：`ImportError`

- [ ] **步骤 3：实现 `biteme/context/factory.py`**

```python
from .base import ContextProvider
from .direct import DirectProvider
from .rag import RAGProvider
from ..indexing.pipeline import estimate_tokens

AUTO_THRESHOLD_TOKENS = 100_000

def create_provider(source_path: str, strategy: str, db_path: str) -> ContextProvider:
    if strategy == "direct":
        return DirectProvider(source_path=source_path)
    if strategy == "rag":
        return RAGProvider(db_path=db_path)
    # strategy == "auto"
    token_count = estimate_tokens(source_path)
    if token_count <= AUTO_THRESHOLD_TOKENS:
        return DirectProvider(source_path=source_path)
    return RAGProvider(db_path=db_path)
```

- [ ] **步骤 4：运行确认通过**

```bash
pytest tests/test_context_factory.py -v
```

预期：`4 passed`

- [ ] **步骤 5：提交**

```bash
git add biteme/context/factory.py tests/test_context_factory.py
git commit -m "feat: 添加 ContextProvider factory（auto/direct/rag 策略）"
```

---

## Task 7：LangGraph 状态与 Prompt 模板

**文件：**
- 创建：`biteme/graph/state.py`
- 创建：`biteme/graph/prompts.py`

- [ ] **步骤 1：实现 `biteme/graph/state.py`**

（无需测试，纯 TypedDict 定义）

```python
from typing import Literal
from typing_extensions import TypedDict

class Turn(TypedDict):
    speaker: str            # "questioner" | "answerer" | "human"
    content: str
    retrieved_chunks: list[str]  # answerer 轮携带，其余为空列表

class SessionState(TypedDict):
    mode: Literal["learn", "interview"]
    messages: list[Turn]
    current_speaker: str           # "questioner" | "answerer"
    hitl_flags: list[str]          # 可包含 "questioner"、"answerer"
    turn_count: int
    max_turns: int
    context_strategy: str          # "auto" | "direct" | "rag"
    source_path: str
```

- [ ] **步骤 2：实现 `biteme/graph/prompts.py`**

```python
LEARN_QUESTIONER = """\
你是一位好奇的学习者，正在探索以下内容。
每次提出一个清晰、具体的问题，帮助自己逐步理解内容的结构、设计与实现。
不要重复已经问过的问题。只输出问题本身，不要加任何前缀或解释。
"""

LEARN_ANSWERER = """\
你是内容专家。基于以下检索到的相关片段，详细回答问题。
如果片段中有文件路径，请在回答中引用它们。
如果检索片段不足以完整回答，请诚实说明。

相关内容：
{context}
"""

INTERVIEW_QUESTIONER = """\
你是一位经验丰富的技术面试官。
每轮提出一个有深度的技术问题，并在问题前简短评价上一轮的回答（第一轮跳过评价）。
只输出评价（若有）+ 问题，不要加其他前缀。
"""

INTERVIEW_ANSWERER = """\
你是一位技术面试候选人。给出简洁、专业的技术回答。
如有必要，可参考以下检索到的相关背景信息，但不要直接照抄。

参考信息：
{context}
"""

def get_prompts(mode: str) -> dict[str, str]:
    if mode == "learn":
        return {"questioner": LEARN_QUESTIONER, "answerer": LEARN_ANSWERER}
    return {"questioner": INTERVIEW_QUESTIONER, "answerer": INTERVIEW_ANSWERER}
```

- [ ] **步骤 3：提交**

```bash
git add biteme/graph/state.py biteme/graph/prompts.py
git commit -m "feat: 添加 LangGraph 状态定义与 prompt 模板"
```

---

## Task 8：LangGraph 节点

**文件：**
- 创建：`biteme/graph/nodes.py`
- 创建：`tests/test_graph_nodes.py`

- [ ] **步骤 1：写失败测试**

```python
# tests/test_graph_nodes.py
import pytest
from unittest.mock import MagicMock, patch
from biteme.graph.state import SessionState, Turn
from biteme.graph.nodes import questioner_node, answerer_node

def make_state(**kwargs) -> SessionState:
    defaults = dict(
        mode="learn",
        messages=[],
        current_speaker="questioner",
        hitl_flags=[],
        turn_count=0,
        max_turns=10,
        context_strategy="direct",
        source_path="/tmp/fake",
    )
    defaults.update(kwargs)
    return defaults  # type: ignore

def test_questioner_node_appends_turn(tmp_path):
    state = make_state(source_path=str(tmp_path))
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "What does foo do?"

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.get_overview.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        result = questioner_node(state)

    mock_provider.get_overview.assert_called_once()  # 第一轮走 get_overview，不走 retrieve
    mock_provider.retrieve.assert_not_called()
    assert result["current_speaker"] == "answerer"
    assert result["turn_count"] == 1
    assert len(result["messages"]) == 1
    assert result["messages"][0]["speaker"] == "questioner"

def test_answerer_node_always_retrieves(tmp_path):
    state = make_state(
        source_path=str(tmp_path),
        current_speaker="answerer",
        messages=[{"speaker": "questioner", "content": "What does foo do?", "retrieved_chunks": []}],
    )
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "foo returns None."

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.retrieve.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        result = answerer_node(state)

    mock_provider.retrieve.assert_called_once()   # 必须调用 retrieve
    assert result["current_speaker"] == "questioner"
    assert result["messages"][-1]["speaker"] == "answerer"
    assert len(result["messages"][-1]["retrieved_chunks"]) > 0
```

- [ ] **步骤 2：运行确认失败**

```bash
pytest tests/test_graph_nodes.py -v
```

预期：`ImportError`

- [ ] **步骤 3：实现 `biteme/graph/nodes.py`**

```python
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
        # 对话历史来自 state["messages"]，与知识库检索无关
        history = "\n".join(
            f"[{t['speaker']}]: {t['content']}" for t in state["messages"][-6:]
        )
        provider = create_provider(
            source_path=state["source_path"],
            strategy=state["context_strategy"],
            db_path=_get_db_path(state["source_path"]),
        )
        # 第一轮：messages 为空，用 get_overview() 让提问者先了解源内容
        # 后续轮：用上轮回答内容作 query，检索相关片段辅助出题
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
    # 对话历史来自 state["messages"]；最后一条问题文本用作知识库检索的 query
    last_question = state["messages"][-1]["content"] if state["messages"] else ""
    provider = create_provider(
        source_path=state["source_path"],
        strategy=state["context_strategy"],
        db_path=_get_db_path(state["source_path"]),
    )
    chunks = provider.retrieve(last_question)  # 必须调用，不可跳过

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
```

- [ ] **步骤 4：运行确认通过**

```bash
pytest tests/test_graph_nodes.py -v
```

预期：`2 passed`

- [ ] **步骤 5：提交**

```bash
git add biteme/graph/nodes.py tests/test_graph_nodes.py
git commit -m "feat: 添加 LangGraph 节点（questioner / answerer）"
```

---

## Task 9：构建 LangGraph 图

**文件：**
- 创建：`biteme/graph/graph.py`

- [ ] **步骤 1：实现 `biteme/graph/graph.py`**

（图结构由节点组合而成，核心逻辑已在节点里测过，这里只做集成）

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from .state import SessionState
from .nodes import questioner_node, answerer_node

def _should_continue(state: SessionState) -> str:
    if state["turn_count"] >= state["max_turns"]:
        return END
    return "continue"

def build_graph(checkpointer: SqliteSaver) -> StateGraph:
    builder = StateGraph(SessionState)
    builder.add_node("questioner", questioner_node)
    builder.add_node("answerer", answerer_node)

    builder.set_entry_point("questioner")
    builder.add_edge("questioner", "answerer")
    builder.add_conditional_edges(
        "answerer",
        _should_continue,
        {"continue": "questioner", END: END},
    )
    return builder.compile(checkpointer=checkpointer, interrupt_before=[])
```

- [ ] **步骤 2：提交**

```bash
git add biteme/graph/graph.py
git commit -m "feat: 构建 LangGraph StateGraph"
```

---

## Task 10：会话管理

**文件：**
- 创建：`biteme/session/manager.py`
- 创建：`tests/test_session_manager.py`

- [ ] **步骤 1：写失败测试**

```python
# tests/test_session_manager.py
import pytest
from pathlib import Path
from biteme.session.manager import create_session, list_sessions, get_checkpoint_saver

def test_create_session_returns_id(tmp_path, monkeypatch):
    monkeypatch.setattr("biteme.session.manager.settings.sessions_dir", tmp_path)
    session_id = create_session(source_path="/tmp/repo", mode="learn")
    assert isinstance(session_id, str)
    assert len(session_id) > 0

def test_list_sessions_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("biteme.session.manager.settings.sessions_dir", tmp_path)
    sessions = list_sessions()
    assert sessions == []

def test_list_sessions_after_create(tmp_path, monkeypatch):
    monkeypatch.setattr("biteme.session.manager.settings.sessions_dir", tmp_path)
    sid = create_session(source_path="/tmp/repo", mode="interview")
    sessions = list_sessions()
    assert any(s["session_id"] == sid for s in sessions)
    assert sessions[0]["mode"] == "interview"
```

- [ ] **步骤 2：运行确认失败**

```bash
pytest tests/test_session_manager.py -v
```

预期：`ImportError`

- [ ] **步骤 3：实现 `biteme/session/manager.py`**

```python
import uuid
import json
from datetime import datetime
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from ..config import settings

def create_session(source_path: str, mode: str) -> str:
    session_id = uuid.uuid4().hex[:8]
    meta = {
        "session_id": session_id,
        "source_path": source_path,
        "mode": mode,
        "created_at": datetime.now().isoformat(),
        "status": "created",
    }
    meta_path = settings.sessions_dir / f"{session_id}.meta.json"
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))
    return session_id

def list_sessions() -> list[dict]:
    sessions_dir = settings.sessions_dir
    if not sessions_dir.exists():
        return []
    sessions = []
    for meta_file in sorted(sessions_dir.glob("*.meta.json"), reverse=True):
        try:
            sessions.append(json.loads(meta_file.read_text()))
        except Exception:
            pass
    return sessions

def get_checkpoint_saver(session_id: str) -> SqliteSaver:
    db_path = settings.sessions_dir / f"{session_id}.db"
    return SqliteSaver.from_conn_string(str(db_path))
```

- [ ] **步骤 4：运行确认通过**

```bash
pytest tests/test_session_manager.py -v
```

预期：`3 passed`

- [ ] **步骤 5：提交**

```bash
git add biteme/session/manager.py tests/test_session_manager.py
git commit -m "feat: 添加会话管理（创建、列表、SQLite checkpoint）"
```

---

## Task 11：CLI

**文件：**
- 创建：`biteme/cli.py`
- 创建：`tests/test_cli.py`

- [ ] **步骤 1：写失败测试**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from biteme.cli import app

runner = CliRunner()

def test_list_empty(tmp_path, monkeypatch):
    with patch("biteme.cli.list_sessions", return_value=[]):
        result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "暂无历史会话" in result.output

def test_index_missing_source():
    result = runner.invoke(app, ["index", "/nonexistent/path"])
    assert result.exit_code != 0
    assert "不存在" in result.output

def test_run_missing_source():
    result = runner.invoke(app, ["run", "/nonexistent/path"])
    assert result.exit_code != 0
    assert "不存在" in result.output
```

- [ ] **步骤 2：运行确认失败**

```bash
pytest tests/test_cli.py -v
```

预期：`ImportError`

- [ ] **步骤 3：实现 `biteme/cli.py`**

```python
from pathlib import Path
from typing import Annotated, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from langgraph.types import Command

from .config import settings
from .indexing.pipeline import build_index
from .session.manager import create_session, list_sessions, get_checkpoint_saver
from .graph.graph import build_graph
from .graph.state import SessionState

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
    turns: Annotated[int, typer.Option(help="最大对话轮数")] = 10,
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
    console.print(Panel(f"会话 ID: [bold]{session_id}[/bold]  模式: [cyan]{mode}[/cyan]  HITL: [yellow]{hitl}[/yellow]", title="BiteMe"))

    initial_state: SessionState = {
        "mode": mode,  # type: ignore
        "messages": [],
        "current_speaker": "questioner",
        "hitl_flags": hitl_flags,
        "turn_count": 0,
        "max_turns": turns,
        "context_strategy": strategy,
        "source_path": str(path.resolve()),
    }

    checkpointer = get_checkpoint_saver(session_id)
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": session_id}}

    _run_graph(graph, initial_state, config)

@app.command()
def resume(
    session_id: Annotated[str, typer.Argument(help="要恢复的会话 ID")],
):
    """恢复上次中断的会话"""
    checkpointer = get_checkpoint_saver(session_id)
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
    table.add_column("ID"), table.add_column("来源"), table.add_column("模式"), table.add_column("时间"), table.add_column("状态")
    for s in sessions:
        table.add_row(s["session_id"], s["source_path"], s["mode"], s["created_at"][:19], s["status"])
    console.print(table)

def _run_graph(graph, initial_state, config):
    """运行图，处理 HITL interrupt 和流式输出。"""
    stream = graph.stream(initial_state, config=config, stream_mode="values") if initial_state else \
             graph.stream(Command(resume=None), config=config, stream_mode="values")

    for state in stream:
        if state.get("messages"):
            last = state["messages"][-1]
            speaker_color = {"questioner": "blue", "answerer": "green", "human": "yellow"}.get(last["speaker"], "white")
            console.print(f"\n[bold {speaker_color}][{last['speaker'].upper()}][/bold {speaker_color}]")
            console.print(last["content"])
            if last.get("retrieved_chunks"):
                console.print(f"[dim]（引用了 {len(last['retrieved_chunks'])} 个片段）[/dim]")
```

- [ ] **步骤 4：运行确认通过**

```bash
pytest tests/test_cli.py -v
```

预期：`3 passed`

- [ ] **步骤 5：运行全量测试确认无回归**

```bash
pytest -v
```

预期：全部通过，无 error

- [ ] **步骤 6：提交**

```bash
git add biteme/cli.py tests/test_cli.py
git commit -m "feat: 添加 CLI（index/run/resume/list）"
```

---

## Task 12：README

**文件：**
- 创建：`README.md`

- [ ] **步骤 1：写 `README.md`**

```markdown
# BiteMe

双 Agent 问答系统。给定任意内容（代码仓库、文档、简历），
一个 Agent 负责提问，一个负责回答，人类可旁观或接管任意一侧。

## 安装

```bash
pip install -e ".[dev]"
cp .env.example .env
# 填入 OPENAI_API_KEY
```

## 快速开始

```bash
# 小文件（自动 direct 策略）
biteme run ./my-resume.md --mode interview --hitl answerer

# 大型代码仓（先建索引）
biteme index ./my-repo
biteme run ./my-repo --mode learn --turns 15

# 查看历史会话
biteme list

# 恢复中断的会话
biteme resume <session-id>
```

## HITL 选项

| `--hitl` | 效果 |
|----------|------|
| `none`（默认）| 纯观察，两侧均为 AI |
| `questioner` | 人类控制提问侧 |
| `answerer` | 人类控制回答侧 |
| `both` | 两侧均由人类输入 |
```

- [ ] **步骤 2：提交**

```bash
git add README.md
git commit -m "docs: 添加 README"
```

---

## 自检结果

- **Spec 覆盖**：ContextProvider 抽象（Task 3/5/6）、建索引（Task 4）、LangGraph 状态与节点（Task 7/8/9）、会话管理（Task 10）、CLI 四个命令（Task 11）、两种模式 prompt（Task 7）全部有对应任务。
- **Placeholder 扫描**：无 TBD/TODO。
- **类型一致性**：`SessionState`、`Turn` 在 `state.py` 定义，所有节点和测试均引用同一定义。`hitl_flags` 全程为 `list[str]`。
