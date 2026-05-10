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
