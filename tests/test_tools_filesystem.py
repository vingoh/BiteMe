import pytest
from pathlib import Path


# ── search_files_by_content ───────────────────────────────────────────────────

class TestSearchFilesByContent:
    def test_finds_matching_line(self, tmp_path):
        from biteme.tools.filesystem import search_files_by_content
        (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
        result = search_files_by_content.invoke({"directory": str(tmp_path), "query": "def foo"})
        assert "a.py:1:" in result
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
        assert "b.txt:3:" in result
        assert "line2" in result
        assert "line4" in result

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


# ── community tools smoke tests ───────────────────────────────────────────────

class TestCommunityToolsExported:
    def test_read_file_exists(self):
        from biteme.tools.filesystem import read_file
        assert hasattr(read_file, "invoke")

    def test_write_file_exists(self):
        from biteme.tools.filesystem import write_file
        assert hasattr(write_file, "invoke")

    def test_search_files_by_name_exists(self):
        from biteme.tools.filesystem import search_files_by_name
        assert hasattr(search_files_by_name, "invoke")

    def test_write_then_read(self, tmp_path):
        from biteme.tools.filesystem import read_file, write_file
        target = str(tmp_path / "out.txt")
        write_file.invoke({"file_path": target, "text": "hello"})
        content = read_file.invoke({"file_path": target})
        assert "hello" in content
