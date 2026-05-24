import re
from pathlib import Path

from langchain_community.tools.file_management import (
    FileSearchTool,
    ReadFileTool,
    WriteFileTool,
)
from langchain_core.tools import tool

# Community tools — no root_dir restriction; agent operates on absolute paths
read_file = ReadFileTool()
write_file = WriteFileTool()
search_files_by_name = FileSearchTool()


@tool
def search_files_by_content(
    directory: str,
    query: str,
    file_glob: str = "*",
    context_lines: int = 2,
) -> str:
    """Search file contents in a directory for lines containing a literal string.

    All queries are treated as plain-text literals (not regex), so characters
    like '$', '.', '*' match themselves. Returns grep-style output:
      path/to/file.py:42: matched line
        40: context before
        43: context after

    Args:
        directory: Root directory to search in.
        query: Plain-text string to match literally.
        file_glob: Glob pattern to filter files (e.g. '*.py'). Defaults to '*'.
        context_lines: Number of lines to show before and after each match.
    """
    root = Path(directory)
    if not root.exists():
        return f"Error: directory '{directory}' does not exist"

    pattern = re.compile(re.escape(query))

    results: list[str] = []
    for filepath in sorted(root.rglob(file_glob)):
        if not filepath.is_file():
            continue
        try:
            lines = filepath.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        rel = filepath.relative_to(root)
        for i, line in enumerate(lines):
            if not pattern.search(line):
                continue
            block = [f"{rel}:{i + 1}: {line}"]
            for j in range(max(0, i - context_lines), i):
                block.append(f"  {j + 1}: {lines[j]}")
            for j in range(i + 1, min(len(lines), i + 1 + context_lines)):
                block.append(f"  {j + 1}: {lines[j]}")
            results.append("\n".join(block))

    if not results:
        return f"No matches found for '{query}' in '{directory}'"
    return "\n\n".join(results)
