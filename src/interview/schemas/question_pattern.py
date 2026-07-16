"""Strategy가 질문 패턴 검색 결과를 소비하는 공용 모델."""

from pydantic import BaseModel, Field


class InterviewQuestionSignal(BaseModel):
    """실제 질문이 아니라 질문 생성에 참고할 패턴 신호."""

    pattern_id: str
    pattern_text: str
    frequency: int = Field(ge=1)
    signal_kind: str
    required_evidence_signals: list[str] = Field(default_factory=list)
    topic_family: str | None = None
    similarity: float = Field(ge=0.0, le=1.0)
