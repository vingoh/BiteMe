# tests/test_config.py
import os
from biteme.config import settings

def test_biteme_home_default(tmp_path, monkeypatch):
    monkeypatch.delenv("BITEME_HOME", raising=False)
    # 重新导入以触发默认值逻辑
    import importlib, biteme.config as cfg
    importlib.reload(cfg)
    assert str(cfg.settings.biteme_home).endswith(".biteme")

def test_biteme_home_custom(tmp_path, monkeypatch):
    monkeypatch.setenv("BITEME_HOME", str(tmp_path))
    import importlib, biteme.config as cfg
    importlib.reload(cfg)
    assert cfg.settings.biteme_home == tmp_path

def test_github_token_default(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    import importlib, biteme.config as cfg
    importlib.reload(cfg)
    assert cfg.settings.github_token == ""

def test_github_token_custom(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
    import importlib, biteme.config as cfg
    importlib.reload(cfg)
    assert cfg.settings.github_token == "ghp_test123"
