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
