import re
from pathlib import Path

from langchain_community.tools.file_management import (
    FileSearchTool,
    WriteFileTool,
)
from langchain_core.tools import tool

write_file = WriteFileTool()
search_files_by_name = FileSearchTool()

_OUTLINE_SUPPORTED = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".md", ".markdown",
}
_OUTLINE_SUPPORTED_STR = " ".join(sorted(_OUTLINE_SUPPORTED))

_MAX_OUTLINE_LINES = 5000


@tool
def read_file(path: str, offset: int = 1, limit: int = 200) -> str:
    """读取本地文件的指定行范围。

    若不确定目标内容在哪一行，建议先调用 file_outline 获取文件结构和行号，
    再用 offset 精准定位，避免读取无关内容浪费 token。

    Args:
        path: 文件的绝对路径。
        offset: 起始行号（从 1 开始）。默认 1。
        limit: 最多读取的行数。默认 200。

    Returns:
        带行号前缀的文件内容，格式：'行号 | 内容'。
    """
    p = Path(path)
    if not p.exists():
        return f"Error: 文件不存在：{path}"
    if not p.is_file():
        return f"Error: 路径不是文件：{path}"
    try:
        lines = p.read_text(errors="ignore").splitlines()
    except Exception as e:
        return f"Error: 无法读取文件：{e}"

    start = max(0, offset - 1)
    end = start + limit
    selected = lines[start:end]
    if not selected:
        return f"（文件共 {len(lines)} 行，offset={offset} 超出范围）"

    return "\n".join(f"{start + i + 1:6} | {line}" for i, line in enumerate(selected))


def _outline_python(lines: list[str]) -> list[str]:
    pattern = re.compile(r'^(\s*)(class|def)\s+(\w+)\s*(\(.*)?')
    results = []
    for i, line in enumerate(lines, 1):
        m = pattern.match(line)
        if m:
            indent = len(m.group(1))
            kind = m.group(2)
            name = m.group(3)
            sig = (m.group(4) or "").rstrip(":")
            prefix = "  " * (indent // 4)
            results.append(f"第{i:5} 行  {prefix}{kind} {name}{sig}")
    return results


def _outline_js_ts(lines: list[str]) -> list[str]:
    patterns = [
        re.compile(r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s+(\w+)\s*\('),
        re.compile(r'^\s*(export\s+)?(abstract\s+)?class\s+(\w+)'),
        re.compile(r'^\s*(export\s+)?const\s+(\w+)\s*=\s*(async\s+)?(\(|function)'),
    ]
    results = []
    for i, line in enumerate(lines, 1):
        for pat in patterns:
            if pat.match(line):
                results.append(f"第{i:5} 行  {line.rstrip()}")
                break
    return results


def _outline_cpp(lines: list[str]) -> list[str]:
    class_pat = re.compile(r'^\s*(class|struct)\s+(\w+)')
    func_pat = re.compile(r'^[\w][\w\s\*\&:<>,~]*\s+[\*\&]*(\w+)\s*\([^;]*$')
    skip_pat = re.compile(r'^\s*(#|//|/\*|\*)')
    results = []
    for i, line in enumerate(lines, 1):
        if skip_pat.match(line):
            continue
        m = class_pat.match(line)
        if m:
            results.append(f"第{i:5} 行  {m.group(1)} {m.group(2)}")
            continue
        if func_pat.match(line):
            results.append(f"第{i:5} 行  {line.rstrip()}")
    return results


def _outline_markdown(lines: list[str]) -> list[str]:
    pattern = re.compile(r'^(#{1,6})\s+(.+)')
    results = []
    for i, line in enumerate(lines, 1):
        if pattern.match(line):
            results.append(f"第{i:5} 行  {line.rstrip()}")
    return results


@tool
def file_outline(path: str) -> str:
    """返回文件的结构骨架：所有类、函数、方法的名称与起始行号。

    仅支持以下文件类型（对其他类型会返回错误，请勿对不支持的文件调用本工具）：
      .py                      — Python
      .js / .ts / .jsx / .tsx  — JavaScript / TypeScript
      .cpp / .cc / .cxx        — C++ 源文件
      .h / .hpp                — C / C++ 头文件
      .md / .markdown          — Markdown（提取各级标题）

    Args:
        path: 文件的绝对路径。

    Returns:
        每行格式：'第 N 行  <缩进>kind 名称(签名)'
        若文件超过 5000 行，只扫描前 5000 行。
    """
    p = Path(path)
    if not p.exists():
        return f"Error: 文件不存在：{path}"
    if not p.is_file():
        return f"Error: 路径不是文件：{path}"

    ext = p.suffix.lower()
    if ext not in _OUTLINE_SUPPORTED:
        return (
            f"file_outline: 不支持的文件类型 '{ext}'，"
            f"支持：{_OUTLINE_SUPPORTED_STR}"
        )

    try:
        all_lines = p.read_text(errors="ignore").splitlines()
    except Exception as e:
        return f"Error: 无法读取文件：{e}"

    truncated = len(all_lines) > _MAX_OUTLINE_LINES
    lines = all_lines[:_MAX_OUTLINE_LINES]

    if ext == ".py":
        entries = _outline_python(lines)
    elif ext in {".js", ".ts", ".jsx", ".tsx"}:
        entries = _outline_js_ts(lines)
    elif ext in {".md", ".markdown"}:
        entries = _outline_markdown(lines)
    else:
        entries = _outline_cpp(lines)

    if not entries:
        return f"（未找到结构定义，共扫描 {len(lines)} 行）"

    result = "\n".join(entries)
    if truncated:
        result += f"\n\n（文件共 {len(all_lines)} 行，仅显示前 {_MAX_OUTLINE_LINES} 行的结构）"
    return result


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
    if not root.is_dir():
        return f"Error: '{directory}' is not a directory or does not exist"

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
            block = []
            for j in range(max(0, i - context_lines), i):
                block.append(f"  {j + 1}: {lines[j]}")
            block.append(f"{rel}:{i + 1}: {line}")
            for j in range(i + 1, min(len(lines), i + 1 + context_lines)):
                block.append(f"  {j + 1}: {lines[j]}")
            results.append("\n".join(block))

    if not results:
        return f"No matches found for '{query}' in '{directory}'"
    return "\n\n".join(results)
