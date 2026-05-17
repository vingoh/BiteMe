from typing import Literal
from typing_extensions import TypedDict


class Turn(TypedDict):
    speaker: str            # "questioner" | "answerer" | "human"
    content: str
    retrieved_chunks: list[str]  # answerer 轮携带，其余为空列表


class SessionState(TypedDict):
    mode: Literal["learn", "interview"]
    messages: list[Turn]
    current_speaker: str           # "questioner" | "answerer"
    hitl_flags: list[str]          # 可包含 "questioner"、"answerer"
    turn_count: int
    max_turns: int
    context_strategy: str          # "auto" | "direct" | "rag"
    source_path: str
    outline: list[str]             # planner 生成的提问大纲，空列表表示尚未生成
