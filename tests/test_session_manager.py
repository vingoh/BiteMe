import pytest
from pathlib import Path
from biteme.session.manager import create_session, list_sessions, get_checkpoint_saver

def test_create_session_returns_id(tmp_path, monkeypatch):
    monkeypatch.setattr("biteme.session.manager.settings.sessions_dir", tmp_path)
    session_id = create_session(source_path="/tmp/repo", mode="learn")
    assert isinstance(session_id, str)
    assert len(session_id) > 0

def test_list_sessions_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("biteme.session.manager.settings.sessions_dir", tmp_path)
    sessions = list_sessions()
    assert sessions == []

def test_list_sessions_after_create(tmp_path, monkeypatch):
    monkeypatch.setattr("biteme.session.manager.settings.sessions_dir", tmp_path)
    sid = create_session(source_path="/tmp/repo", mode="interview")
    sessions = list_sessions()
    assert any(s["session_id"] == sid for s in sessions)
    assert sessions[0]["mode"] == "interview"
