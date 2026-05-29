import pytest
from pathlib import Path


# ── search_files_by_content ───────────────────────────────────────────────────

class TestSearchFilesByContent:
    def test_finds_matching_line(self, tmp_path):
        from biteme.tools.filesystem import search_files_by_content
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
        result = search_files_by_content.invoke({"directory": str(tmp_path), "query": "def foo"})
        assert "a.py" in result
        assert ">    1 |" in result
        assert "def foo" in result

    def test_includes_context_lines(self, tmp_path):
        from biteme.tools.filesystem import search_files_by_content
        lines = ["line1\n", "line2\n", "MATCH\n", "line4\n", "line5\n"]
        (tmp_path / "b.txt").write_text("".join(lines))
        result = search_files_by_content.invoke({
            "directory": str(tmp_path),
            "query": "MATCH",
            "context_lines": 1,
        })
        assert "b.txt" in result
        assert ">    3 |" in result
        assert "line2" in result
        assert "line4" in result
        # Verify chronological order: line2 before match, match before line4
        assert result.index("line2") < result.index("MATCH") < result.index("line4")

    def test_file_glob_filters(self, tmp_path):
        from biteme.tools.filesystem import search_files_by_content
        (tmp_path / "a.py").write_text("needle\n")
        (tmp_path / "b.md").write_text("needle\n")
        result = search_files_by_content.invoke({
            "directory": str(tmp_path),
            "query": "needle",
            "file_glob": "*.py",
        })
        assert "a.py" in result
        assert "b.md" not in result

    def test_no_match_returns_informative_string(self, tmp_path):
        from biteme.tools.filesystem import search_files_by_content
        (tmp_path / "empty.py").write_text("nothing here\n")
        result = search_files_by_content.invoke({"directory": str(tmp_path), "query": "xyzzy_not_found"})
        assert "No matches" in result

    def test_nonexistent_directory_returns_error(self):
        from biteme.tools.filesystem import search_files_by_content
        result = search_files_by_content.invoke({"directory": "/no/such/dir", "query": "foo"})
        assert result.startswith("Error:")

    def test_plain_text_query_not_treated_as_broken_regex(self, tmp_path):
        from biteme.tools.filesystem import search_files_by_content
        (tmp_path / "f.py").write_text("price: $100\n")
        # "$100" contains regex special chars — should not raise
        result = search_files_by_content.invoke({"directory": str(tmp_path), "query": "$100"})
        assert "f.py" in result


# ── tools smoke tests ─────────────────────────────────────────────────────────

class TestToolsExported:
    def test_read_file_exists(self):
        from biteme.tools.filesystem import read_file
        assert hasattr(read_file, "invoke")

    def test_write_file_exists(self):
        from biteme.tools.filesystem import write_file
        assert hasattr(write_file, "invoke")

    def test_search_files_by_name_exists(self):
        from biteme.tools.filesystem import search_files_by_name
        assert hasattr(search_files_by_name, "invoke")

    def test_list_directory_exists(self):
        from biteme.tools.filesystem import list_directory
        assert hasattr(list_directory, "invoke")

    def test_write_then_read(self, tmp_path):
        from biteme.tools.filesystem import read_file, write_file
        target = str(tmp_path / "out.txt")
        write_file.invoke({"file_path": target, "text": "hello"})
        content = read_file.invoke({"path": target})
        assert "hello" in content


# ── search_files_by_name ──────────────────────────────────────────────────────

class TestSearchFilesByName:
    def test_finds_py_files(self, tmp_path):
        from biteme.tools.filesystem import search_files_by_name
        (tmp_path / "foo.py").write_text("")
        (tmp_path / "bar.txt").write_text("")
        result = search_files_by_name.invoke({"directory": str(tmp_path), "pattern": "*.py"})
        assert "foo.py" in result
        assert "bar.txt" not in result

    def test_returns_relative_paths(self, tmp_path):
        from biteme.tools.filesystem import search_files_by_name
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("")
        result = search_files_by_name.invoke({"directory": str(tmp_path), "pattern": "*.py"})
        assert "sub" in result
        assert "nested.py" in result
        # Must not contain the absolute tmp_path prefix
        assert str(tmp_path) not in result

    def test_no_match_returns_informative_string(self, tmp_path):
        from biteme.tools.filesystem import search_files_by_name
        result = search_files_by_name.invoke({"directory": str(tmp_path), "pattern": "*.go"})
        assert "未找到" in result

    def test_nonexistent_directory_returns_error(self):
        from biteme.tools.filesystem import search_files_by_name
        result = search_files_by_name.invoke({"directory": "/no/such/dir", "pattern": "*.py"})
        assert result.startswith("Error:")


# ── list_directory ────────────────────────────────────────────────────────────

class TestListDirectory:
    def test_lists_files_and_dirs(self, tmp_path):
        from biteme.tools.filesystem import list_directory
        (tmp_path / "readme.md").write_text("# hello\n")
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("pass\n")
        result = list_directory.invoke({"path": str(tmp_path)})
        assert "readme.md" in result
        assert "src" in result
        assert "main.py" in result

    def test_depth_limits_expansion(self, tmp_path):
        from biteme.tools.filesystem import list_directory
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("")
        # depth=1: only show 'a/', its contents should be collapsed
        result = list_directory.invoke({"path": str(tmp_path), "depth": 1})
        assert "a/" in result
        assert "deep.py" not in result
        assert "items" in result  # collapsed "(N items)" indicator

    def test_skips_noise_dirs(self, tmp_path):
        from biteme.tools.filesystem import list_directory
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / ".git").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "real.py").write_text("")
        result = list_directory.invoke({"path": str(tmp_path)})
        assert "__pycache__" not in result
        assert ".git" not in result
        assert "node_modules" not in result
        assert "real.py" in result

    def test_file_path_returns_single_file_info(self, tmp_path):
        from biteme.tools.filesystem import list_directory
        f = tmp_path / "single.py"
        f.write_text("line1\nline2\n")
        result = list_directory.invoke({"path": str(f)})
        assert "单文件" in result
        assert "single.py" in result

    def test_nonexistent_path_returns_error(self):
        from biteme.tools.filesystem import list_directory
        result = list_directory.invoke({"path": "/no/such/path"})
        assert result.startswith("Error:")
