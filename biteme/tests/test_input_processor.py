from pathlib import Path

import pytest

from src.input.processor import build_project_content


def test_build_project_content_from_text() -> None:
    content = build_project_content(raw_text="Hello BiteMe", folder_path=None)
    assert content == "Hello BiteMe"


def test_build_project_content_from_explicit_empty_text() -> None:
    content = build_project_content(raw_text="", folder_path=None)
    assert content == ""


def test_build_project_content_from_folder_recursive(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    nested = tmp_path / "nested" / "pkg"
    nested.mkdir(parents=True)
    (nested / "core.txt").write_text("core details", encoding="utf-8")

    content = build_project_content(raw_text=None, folder_path=str(tmp_path))

    assert "README.md" in content
    assert "app.py" in content
    assert "nested/pkg/core.txt" in content


def test_build_project_content_ignores_git_directory(tmp_path: Path) -> None:
    (tmp_path / "main.txt").write_text("keep me", encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("drop me", encoding="utf-8")

    content = build_project_content(raw_text=None, folder_path=str(tmp_path))

    assert "main.txt" in content
    assert ".git/config" not in content
    assert "drop me" not in content


def test_build_project_content_raises_for_invalid_folder_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError, match="Folder does not exist"):
        build_project_content(raw_text=None, folder_path=str(missing_path))


def test_build_project_content_raises_for_non_directory_path(tmp_path: Path) -> None:
    file_path = tmp_path / "single.txt"
    file_path.write_text("not a folder", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="Path is not a directory"):
        build_project_content(raw_text=None, folder_path=str(file_path))


def test_build_project_content_raises_when_no_input() -> None:
    with pytest.raises(ValueError):
        build_project_content(raw_text=None, folder_path=None)
