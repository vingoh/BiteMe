from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_rounds: int = 8
