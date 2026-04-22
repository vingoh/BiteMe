from pathlib import Path


def read_project_folder(folder_path: str) -> str:
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    blocks: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue

        relative_path = path.relative_to(root)
        content = path.read_text(encoding="utf-8", errors="ignore")
        blocks.append(f"## FILE: {relative_path}\n{content}\n")

    return "\n".join(blocks)
