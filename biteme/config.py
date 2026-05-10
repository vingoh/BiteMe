from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    biteme_home: Path = Path(os.getenv("BITEME_HOME", "~/.biteme")).expanduser()
    indexes_dir: Path = property(lambda self: self.biteme_home / "indexes")
    sessions_dir: Path = property(lambda self: self.biteme_home / "sessions")

    def ensure_dirs(self) -> None:
        self.biteme_home.mkdir(parents=True, exist_ok=True)
        (self.biteme_home / "indexes").mkdir(exist_ok=True)
        (self.biteme_home / "sessions").mkdir(exist_ok=True)

settings = Settings()
