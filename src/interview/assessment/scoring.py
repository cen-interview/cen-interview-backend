"""질문 세트 단위 점수 산정 로직.

메인 질문 답변과 후속 질문 답변을 묶어 하나의 점수를 계산한다.
"""

from pydantic import BaseModel, Field

from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


class AnswerAttempt(BaseModel):
    """하나의 질문에 대한 답변 시도."""
    question_id: str
    question_text: str
    answer_text: str
    signal: AnswerQualitySignal


class QuestionSetScore(BaseModel):
    """메인 질문 + 후속 질문을 합산한 최종 점수."""
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
    """메인 답변과 후속 답변들을 합쳐 최종 점수를 계산한다.

    현재는 실제 채점 전 임시 규칙이다.
    """

    if not attempts:
        return QuestionSetScore(
            score=0.0,
            accuracy=0.0,
            sufficiency=0.0,
            final_quality=AnswerQuality.CONFIRM_NEGATIVE,
            strengths=[],
            improvements=["답변 기록이 없습니다."],
            comment="답변 기록이 없어 평가할 수 없습니다.",
        )

    qualities = [attempt.signal.quality for attempt in attempts]

    if AnswerQuality.MISCONCEPTION in qualities:
        return QuestionSetScore(
            score=40.0,
            accuracy=0.3,
            sufficiency=0.4,
            final_quality=AnswerQuality.MISCONCEPTION,
            strengths=["답변을 시도했으나 일부 개념 이해가 필요합니다."],
            improvements=["오개념으로 판단된 부분을 다시 정리해야 합니다."],
            comment="오개념 또는 논리적 허점이 포함되어 낮은 점수로 평가했습니다.",
        )

    if AnswerQuality.CONFIRM_NEGATIVE in qualities:
        return QuestionSetScore(
            score=50.0,
            accuracy=0.5,
            sufficiency=0.5,
            final_quality=AnswerQuality.CONFIRM_NEGATIVE,
            strengths=["질문에 대한 답변은 제시했습니다."],
            improvements=["Evidence 또는 이전 답변과 충돌하는 내용을 명확히 정리해야 합니다."],
            comment="근거 또는 이전 답변과 충돌하는 내용이 있어 확인이 필요합니다.",
        )

    if AnswerQuality.TRAP_AVAILABLE in qualities:
        return QuestionSetScore(
            score=60.0,
            accuracy=0.6,
            sufficiency=0.6,
            final_quality=AnswerQuality.TRAP_AVAILABLE,
            strengths=["기본 답변 흐름은 유지했습니다."],
            improvements=["헷갈리기 쉬운 개념 간 차이를 더 명확히 설명해야 합니다."],
            comment="개념 구분 확인이 필요한 답변으로 평가했습니다.",
        )

    if qualities[-1] == AnswerQuality.SUFFICIENT:
        return QuestionSetScore(
            score=80.0,
            accuracy=0.8,
            sufficiency=0.8,
            final_quality=AnswerQuality.SUFFICIENT,
            strengths=["후속 질문을 통해 부족한 내용을 보완했습니다."],
            improvements=["초기 답변에서 핵심 내용을 더 충분히 설명하면 좋습니다."],
            comment="후속 답변까지 반영하여 충분한 답변으로 평가했습니다.",
        )

    return QuestionSetScore(
        score=65.0,
        accuracy=0.6,
        sufficiency=0.5,
        final_quality=AnswerQuality.BONUS_AVAILABLE,
        strengths=["기본 개념에 대한 이해는 일부 확인되었습니다."],
        improvements=["원인, 사례, 한계점 등 추가 설명이 필요합니다."],
        comment="답변은 가능했지만 추가 확인할 요소가 남아 있습니다.",
    )