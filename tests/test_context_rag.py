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
