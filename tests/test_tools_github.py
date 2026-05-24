import json
import pytest
from unittest.mock import patch, MagicMock


# ── helpers ──────────────────────────────────────────────────────────────────

def _mock_response(status_code: int, json_data):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    if status_code >= 400:
        from requests import HTTPError
        mock.raise_for_status.side_effect = HTTPError(response=mock)
    else:
        mock.raise_for_status.return_value = None
    return mock


# ── github_list_tree ──────────────────────────────────────────────────────────

class TestGithubListTree:
    def test_returns_file_list(self):
        from biteme.tools.github import github_list_tree
        payload = [
            {"name": "README.md", "type": "file", "path": "README.md"},
            {"name": "src",       "type": "dir",  "path": "src"},
        ]
        with patch("biteme.tools.github.requests.get", return_value=_mock_response(200, payload)):
            result = github_list_tree.invoke({"repo_url": "owner/repo"})
        items = json.loads(result)
        assert len(items) == 2
        assert items[0]["name"] == "README.md"
        assert items[1]["type"] == "dir"

    def test_accepts_full_url(self):
        from biteme.tools.github import github_list_tree
        with patch("biteme.tools.github.requests.get", return_value=_mock_response(200, [])) as mock_get:
            github_list_tree.invoke({"repo_url": "https://github.com/owner/repo"})
        url = mock_get.call_args[0][0]
        assert "owner/repo" in url
        assert "github.com/owner/repo" not in url  # resolved to API path

    def test_returns_error_string_on_404(self):
        from biteme.tools.github import github_list_tree
        with patch("biteme.tools.github.requests.get", return_value=_mock_response(404, {})):
            result = github_list_tree.invoke({"repo_url": "owner/repo"})
        assert result.startswith("Error:")


# ── github_read_file ──────────────────────────────────────────────────────────

class TestGithubReadFile:
    def test_decodes_base64_content(self):
        import base64
        from biteme.tools.github import github_read_file
        raw = "hello world"
        encoded = base64.b64encode(raw.encode()).decode() + "\n"
        payload = {"encoding": "base64", "content": encoded}
        with patch("biteme.tools.github.requests.get", return_value=_mock_response(200, payload)):
            result = github_read_file.invoke({"repo_url": "owner/repo", "file_path": "hello.txt"})
        assert result == raw

    def test_returns_error_on_404(self):
        from biteme.tools.github import github_read_file
        with patch("biteme.tools.github.requests.get", return_value=_mock_response(404, {})):
            result = github_read_file.invoke({"repo_url": "owner/repo", "file_path": "missing.txt"})
        assert result.startswith("Error:")


# ── github_search_code ────────────────────────────────────────────────────────

class TestGithubSearchCode:
    def test_returns_search_results(self):
        from biteme.tools.github import github_search_code
        payload = {
            "items": [
                {
                    "path": "src/main.py",
                    "html_url": "https://github.com/owner/repo/blob/main/src/main.py",
                    "text_matches": [{"fragment": "def hello_world():"}],
                }
            ]
        }
        with patch("biteme.tools.github.requests.get", return_value=_mock_response(200, payload)):
            result = github_search_code.invoke({"repo_url": "owner/repo", "query": "hello_world"})
        items = json.loads(result)
        assert len(items) == 1
        assert items[0]["path"] == "src/main.py"
        assert "hello_world" in items[0]["fragment"]

    def test_returns_error_on_api_failure(self):
        from biteme.tools.github import github_search_code
        with patch("biteme.tools.github.requests.get", return_value=_mock_response(422, {})):
            result = github_search_code.invoke({"repo_url": "owner/repo", "query": "foo"})
        assert result.startswith("Error:")
