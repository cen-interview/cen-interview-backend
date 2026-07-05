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

    # 사용자 답변
    answer: str

    # 질문 세트 최종 품질
    quality: AnswerQuality

    # 점수
    score: float = Field(ge=0.0, le=100.0)

    # 세부 점수
    accuracy: float = Field(ge=0.0, le=1.0)
    sufficiency: float = Field(ge=0.0, le=1.0)

    # 질문 세트 평가
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)

    # 최종 코멘트
    comment: str | None = None

    # 음성 전용
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
