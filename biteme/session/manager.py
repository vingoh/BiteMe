import uuid
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from ..config import settings

def create_session(source_path: str, mode: str) -> str:
    session_id = uuid.uuid4().hex[:8]
    meta = {
        "session_id": session_id,
        "source_path": source_path,
        "mode": mode,
        "created_at": datetime.now().isoformat(),
        "status": "created",
    }
    meta_path = settings.sessions_dir / f"{session_id}.meta.json"
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))
    return session_id

def list_sessions() -> list[dict]:
    sessions_dir = settings.sessions_dir
    if not sessions_dir.exists():
        return []
    sessions = []
    for meta_file in sorted(sessions_dir.glob("*.meta.json"), reverse=True):
        try:
            sessions.append(json.loads(meta_file.read_text()))
        except Exception:
            pass
    return sessions

@contextmanager
def get_checkpoint_saver(session_id: str):
    db_path = settings.sessions_dir / f"{session_id}.db"
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver
