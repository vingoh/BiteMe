import base64
import json
import re

import requests
from langchain_core.tools import tool

from ..config import settings


def _parse_repo(repo_url: str) -> str:
    """Normalize 'https://github.com/owner/repo' or 'owner/repo' → 'owner/repo'."""
    repo_url = repo_url.strip().rstrip("/")
    match = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?$", repo_url)
    if match:
        return match.group(1)
    return repo_url


def _headers(accept: str = "application/vnd.github+json") -> dict:
    h = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    if settings.github_token:
        h["Authorization"] = f"Bearer {settings.github_token}"
    return h


@tool
def github_list_tree(repo_url: str, path: str = "", ref: str = "HEAD") -> str:
    """List files and directories at a path in a GitHub repository.

    Returns a JSON array where each item has 'name', 'type' (file/dir), and 'path'.
    repo_url accepts 'https://github.com/owner/repo' or 'owner/repo'.
    """
    repo = _parse_repo(repo_url)
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        r = requests.get(url, headers=_headers(), params={"ref": ref}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            items = [
                {"name": item["name"], "type": "dir" if item["type"] == "dir" else "file", "path": item["path"]}
                for item in data
            ]
        else:
            items = [{"name": data["name"], "type": data.get("type", "file"), "path": data["path"]}]
        return json.dumps(items, ensure_ascii=False)
    except requests.HTTPError as e:
        return f"Error: GitHub API returned {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error: {e}"


@tool
def github_read_file(repo_url: str, file_path: str, ref: str = "HEAD") -> str:
    """Read the raw content of a single file from a GitHub repository.

    Returns the file content as a UTF-8 string.
    repo_url accepts 'https://github.com/owner/repo' or 'owner/repo'.
    """
    repo = _parse_repo(repo_url)
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    try:
        r = requests.get(url, headers=_headers(), params={"ref": ref}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return data.get("content", "")
    except requests.HTTPError as e:
        return f"Error: GitHub API returned {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error: {e}"


@tool
def github_search_code(repo_url: str, query: str, max_results: int = 20) -> str:
    """Search for code matching a keyword in a GitHub repository.

    Returns a JSON array with up to max_results items, each containing 'path', 'url', 'fragment'.
    repo_url accepts 'https://github.com/owner/repo' or 'owner/repo'.
    """
    repo = _parse_repo(repo_url)
    url = "https://api.github.com/search/code"
    params = {"q": f"{query} repo:{repo}", "per_page": min(max_results, 100)}
    accept = "application/vnd.github.text-match+json"
    try:
        r = requests.get(url, headers=_headers(accept), params=params, timeout=10)
        r.raise_for_status()
        raw_items = r.json().get("items", [])[:max_results]
        items = []
        for item in raw_items:
            fragment = ""
            if item.get("text_matches"):
                fragment = item["text_matches"][0].get("fragment", "")
            items.append({"path": item["path"], "url": item["html_url"], "fragment": fragment})
        return json.dumps(items, ensure_ascii=False)
    except requests.HTTPError as e:
        return f"Error: GitHub API returned {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error: {e}"
