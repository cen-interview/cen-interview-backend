"""면접 세션 상태.

한 번의 면접 동안 유지되는 모든 상태. LangGraph 의 그래프 상태로도 쓰인다.
"""

from pydantic import BaseModel, Field

from interview.schemas.events import Mode
from interview.schemas.question import Question
from interview.schemas.report import AnswerEvaluation, CompetencyModel


class SessionState(BaseModel):
    session_id: str
    mode: Mode
    max_questions: int = 10

    current_question: Question | None = None
    asked_count: int = 0

    # 누적 평가
    evaluations: list[AnswerEvaluation] = Field(default_factory=list)
    competency: CompetencyModel = Field(default_factory=CompetencyModel)

    finished: bool = False

    def is_done(self) -> bool:
        """정해진 질문 수에 도달했는지."""
        return self.finished or self.asked_count >= self.max_questions
