import os

# TavilySearch validates TAVILY_API_KEY at import time;
# provide a dummy key so the tool can be instantiated in tests.
os.environ.setdefault("TAVILY_API_KEY", "test-key-not-real")


def test_tavily_search_is_importable():
    from biteme.tools.web import tavily_search
    assert tavily_search is not None
    assert tavily_search.name == "tavily_search"


def test_tavily_search_in_readonly_tools():
    from biteme.tools import READONLY_TOOLS
    names = [t.name for t in READONLY_TOOLS]
    assert "tavily_search" in names


def test_write_file_not_in_readonly_tools():
    from biteme.tools import READONLY_TOOLS
    names = [t.name for t in READONLY_TOOLS]
    assert "write_file" not in names
