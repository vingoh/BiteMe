from dataclasses import dataclass, field


@dataclass
class PlannerMemory:
    last_topics: list[str] = field(default_factory=list)


@dataclass
class QuestionerMemory:
    last_question_id: str | None = None


@dataclass
class AnswererMemory:
    last_answer: str | None = None
