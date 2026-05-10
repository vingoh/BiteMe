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
