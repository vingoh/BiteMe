from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    def __init__(self) -> None:
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.biteme_home: Path = Path(os.getenv("BITEME_HOME", "~/.biteme")).expanduser()
        self.indexes_dir: Path = self.biteme_home / "indexes"
        self.sessions_dir: Path = self.biteme_home / "sessions"

    def ensure_dirs(self) -> None:
        self.biteme_home.mkdir(parents=True, exist_ok=True)
        (self.biteme_home / "indexes").mkdir(exist_ok=True)
        (self.biteme_home / "sessions").mkdir(exist_ok=True)

settings = Settings()
