from pathlib import Path
from .base import ContextProvider

_SUPPORTED_EXTENSIONS = {".py", ".md", ".txt", ".ts", ".js", ".go", ".rs", ".java", ".yaml", ".toml", ".json"}

class DirectProvider(ContextProvider):
    def __init__(self, source_path: str) -> None:
        self._path = Path(source_path)

    def _read_all(self) -> list[str]:
        if self._path.is_file():
            return [self._path.read_text(errors="ignore")]
        chunks = []
        for f in sorted(self._path.rglob("*")):
            if f.is_file() and f.suffix in _SUPPORTED_EXTENSIONS:
                chunks.append(f.read_text(errors="ignore"))
        return chunks

    def get_overview(self) -> list[str]:
        return self._read_all()

    def retrieve(self, query: str) -> list[str]:
        # DirectProvider 内容全量在内存里，query 不影响结果
        return self._read_all()
