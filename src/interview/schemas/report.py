"""
AnswerEvaluation / CompetencyModel / FinalReport

Assessment는 메인 질문과 후속 질문을 하나의 질문 세트로 묶어 평가한다.
면접 중 AnswerEvaluation을 누적하고, 종료 시 FinalReport를 생성한다.
"""


from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# signals.py 의 quality 값을 재사용한다 (같은 4종을 일관되게 쓰기 위함)
from .signals import AnswerQuality


class AnswerEvaluation(BaseModel):
    # 질문 정보
    question_id: str
    topic: str
    question: str

    # 메인 질문의 첫 답변
    answer: str

    # 질문 세트에 포함된 답변 ID
    answer_ids: list[str] = Field(default_factory=list)

    # 메인 질문에서 파생된 질문 ID
    derived_question_ids: list[str] = Field(default_factory=list)

    # 질문 세트 최종 품질
    quality: AnswerQuality

    score: float = Field(ge=0.0, le=100.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    sufficiency: float = Field(ge=0.0, le=1.0)

    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)

    comment: str | None = None
    delivery_note: str | None = None


class CompetencyModel(BaseModel):
    """면접 내내 누적되는 역량 상태."""

    topic_scores: dict[str, float] = Field(default_factory=dict)

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    average_score: float = 0


class FinalReport(BaseModel):

    overall_score: float

    summary: str

    strengths: list[str]

    weaknesses: list[str]

    topics_to_improve: list[str]

    learning_recommendations: list[str]

    evaluations: list[AnswerEvaluation]
