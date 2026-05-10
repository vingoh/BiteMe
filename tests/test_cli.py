from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from biteme.cli import app

runner = CliRunner()

def test_list_empty(tmp_path, monkeypatch):
    with patch("biteme.cli.list_sessions", return_value=[]):
        result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "暂无历史会话" in result.output

def test_index_missing_source():
    result = runner.invoke(app, ["index", "/nonexistent/path"])
    assert result.exit_code != 0
    assert "不存在" in result.output

def test_run_missing_source():
    result = runner.invoke(app, ["run", "/nonexistent/path"])
    assert result.exit_code != 0
    assert "不存在" in result.output
