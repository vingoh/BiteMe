from typing import Optional

from src.input.file_reader import read_project_folder


def build_project_content(*, raw_text: Optional[str], folder_path: Optional[str]) -> str:
    if raw_text is not None:
        return raw_text

    if folder_path is not None:
        return read_project_folder(folder_path)

    raise ValueError("Either raw_text or folder_path is required")
