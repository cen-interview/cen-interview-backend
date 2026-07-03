"""질문 세트 단위 점수 산정 로직.

메인 질문과 파생 질문의 답변을 묶어 하나의 점수를 계산한다.
현재는 실제 채점 규칙 구현 전이므로 임시 점수를 반환한다.
"""

from pydantic import BaseModel, Field

from interview.schemas.question import QuestionKind
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


class AnswerAttempt(BaseModel):
    """질문 하나에 대한 답변 및 평가 기록."""

    answer_id: str

    question_id: str
    question_text: str
    question_kind: QuestionKind

    answer_text: str
    signal: AnswerQualitySignal

    delivery_metrics: dict | None = None


class QuestionSetScore(BaseModel):
    """메인 질문과 파생 질문을 합산한 최종 점수."""

    score: float = Field(ge=0.0, le=100.0)

    accuracy: float = Field(ge=0.0, le=1.0)
    sufficiency: float = Field(ge=0.0, le=1.0)

    final_quality: AnswerQuality

    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)

    comment: str | None = None


def score_question_set(
    attempts: list[AnswerAttempt],
) -> QuestionSetScore:
    """질문 세트에 포함된 모든 답변을 이용해 최종 점수를 계산한다."""

    if not attempts:
        return _empty_question_set_score()

    return _temporary_score_question_set(attempts)


def _empty_question_set_score() -> QuestionSetScore:
    """답변 기록이 없을 때 반환하는 점수."""

    return QuestionSetScore(
        score=0.0,
        accuracy=0.0,
        sufficiency=0.0,
        final_quality=AnswerQuality.CONFIRM_NEGATIVE,
        strengths=[],
        improvements=["답변 기록 없음"],
        comment="답변 기록이 없어 평가할 수 없습니다.",
    )


def _temporary_score_question_set(
    attempts: list[AnswerAttempt],
) -> QuestionSetScore:
    """실제 채점 규칙 구현 전 사용하는 임시 점수."""

    first_attempt = attempts[0]
    last_attempt = attempts[-1]

    first_quality = first_attempt.signal.quality
    final_quality = last_attempt.signal.quality

    all_rationale = [
        rationale
        for attempt in attempts
        for rationale in attempt.signal.rationale
    ]

    # 첫 답변이 충분했던 경우
    if (
        len(attempts) == 1
        and final_quality == AnswerQuality.SUFFICIENT
    ):
        return QuestionSetScore(
            score=90.0,
            accuracy=0.9,
            sufficiency=0.9,
            final_quality=AnswerQuality.SUFFICIENT,
            strengths=all_rationale or ["핵심 내용 설명"],
            improvements=[],
            comment="첫 답변에서 충분한 내용을 설명했습니다.",
        )

    # 첫 답변은 부족했지만 파생 질문에서 보완한 경우
    if (
        len(attempts) > 1
        and final_quality == AnswerQuality.SUFFICIENT
    ):
        return QuestionSetScore(
            score=80.0,
            accuracy=0.8,
            sufficiency=0.8,
            final_quality=AnswerQuality.SUFFICIENT,
            strengths=[
                "파생 질문을 통한 답변 보완",
            ],
            improvements=first_attempt.signal.rationale,
            comment="초기 답변의 부족한 부분을 파생 질문에서 보완했습니다.",
        )

    if final_quality == AnswerQuality.MISCONCEPTION:
        return QuestionSetScore(
            score=40.0,
            accuracy=0.3,
            sufficiency=0.4,
            final_quality=AnswerQuality.MISCONCEPTION,
            strengths=[],
            improvements=last_attempt.signal.rationale,
            comment="답변에 오개념 또는 논리적 문제가 남아 있습니다.",
        )

    if final_quality == AnswerQuality.CONFIRM_NEGATIVE:
        return QuestionSetScore(
            score=50.0,
            accuracy=0.5,
            sufficiency=0.5,
            final_quality=AnswerQuality.CONFIRM_NEGATIVE,
            strengths=[],
            improvements=last_attempt.signal.rationale,
            comment="근거 또는 이전 답변과 충돌하는 내용이 남아 있습니다.",
        )

    if final_quality == AnswerQuality.TRAP_AVAILABLE:
        return QuestionSetScore(
            score=60.0,
            accuracy=0.6,
            sufficiency=0.6,
            final_quality=AnswerQuality.TRAP_AVAILABLE,
            strengths=[],
            improvements=last_attempt.signal.rationale,
            comment="유사한 개념을 구분하는 추가 확인이 필요합니다.",
        )

    if final_quality == AnswerQuality.CONFIRM_POSITIVE:
        return QuestionSetScore(
            score=70.0,
            accuracy=0.7,
            sufficiency=0.7,
            final_quality=AnswerQuality.CONFIRM_POSITIVE,
            strengths=last_attempt.signal.rationale,
            improvements=["적용 범위 또는 사실관계 확인 필요"],
            comment="답변은 대체로 맞지만 추가 확인이 필요합니다.",
        )

    return QuestionSetScore(
        score=65.0,
        accuracy=0.6,
        sufficiency=0.5,
        final_quality=AnswerQuality.BONUS_AVAILABLE,
        strengths=[],
        improvements=last_attempt.signal.rationale,
        comment="답변은 대체로 맞지만 추가 설명이 필요합니다.",
    )