"""질문 세트 단위 점수 산정 로직.

메인 질문 답변과 후속 질문 답변을 묶어 하나의 점수를 계산한다.
"""

from pydantic import BaseModel, Field

from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


class AnswerAttempt(BaseModel):
    """하나의 질문에 대한 답변 시도."""
    question_id: str
    answer_text: str
    signal: AnswerQualitySignal


class QuestionSetScore(BaseModel):
    """메인 질문 + 후속 질문을 합산한 최종 점수."""
    accuracy: float = Field(ge=0.0, le=1.0)
    sufficiency: float = Field(ge=0.0, le=1.0)
    final_quality: AnswerQuality
    comment: str | None = None


def score_question_set(
    attempts: list[AnswerAttempt],
) -> QuestionSetScore:
    """메인 답변과 후속 답변들을 합쳐 최종 점수를 계산한다.

    현재는 실제 채점 전 임시 규칙이다.
    """

    if not attempts:
        return QuestionSetScore(
            accuracy=0.0,
            sufficiency=0.0,
            final_quality=AnswerQuality.SHALLOW,
            comment="답변 기록이 없습니다.",
        )

    qualities = [attempt.signal.quality for attempt in attempts]

    if AnswerQuality.MISCONCEPTION in qualities:
        return QuestionSetScore(
            accuracy=0.3,
            sufficiency=0.4,
            final_quality=AnswerQuality.MISCONCEPTION,
            comment="오개념이 포함되어 낮은 점수로 평가했습니다.",
        )

    if qualities[-1] == AnswerQuality.SUFFICIENT:
        return QuestionSetScore(
            accuracy=0.8,
            sufficiency=0.8,
            final_quality=AnswerQuality.SUFFICIENT,
            comment="후속 답변까지 반영하여 충분한 답변으로 평가했습니다.",
        )

    return QuestionSetScore(
        accuracy=0.5,
        sufficiency=0.4,
        final_quality=AnswerQuality.SHALLOW,
        comment="후속 질문 후에도 핵심 개념 설명이 부족합니다.",
    )