"""
conftest.py – test fixtures and global patches.

tiktoken's cl100k_base encoding is fetched from openaipublic.blob.core.windows.net
on first use.  In offline / proxied CI environments that host is unreachable, so
we patch tiktoken.get_encoding at the session level to avoid any network access.
"""
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def _patch_tiktoken(monkeypatch):
    """Replace tiktoken.get_encoding with a lightweight offline stub."""
    mock_enc = MagicMock()
    # encode() must return something with a len(); simulate ~4 chars per token
    mock_enc.encode.side_effect = lambda text, **kw: list(range(max(1, len(text) // 4)))
    monkeypatch.setattr("tiktoken.get_encoding", lambda *a, **kw: mock_enc)


def pytest_collection_modifyitems(config, items):
    """Skip tests marked 'integration' unless -m integration is explicitly passed."""
    mark_expr = config.option.markexpr if hasattr(config.option, "markexpr") else ""
    if "integration" in mark_expr:
        return  # user opted in — run them normally
    skip = pytest.mark.skip(reason="integration test: run with -m integration")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
