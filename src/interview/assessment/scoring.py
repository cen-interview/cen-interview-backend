"""질문 세트 단위 점수 산정 로직.

메인 질문과 파생 질문의 답변을 묶어 하나의 문항 점수를 계산한다.

이 파일의 역할:
  1. 질문 1개에 대한 답변 시도 정보를 AnswerAttempt로 저장한다.
  2. 메인 질문 + 파생 질문 답변들을 하나의 질문 세트로 보고 점수를 계산한다.
  3. FinalReport에 들어갈 문항별 점수(score)와 평가 코멘트(comment)를 생성한다.

주의:
  - 답변 하나의 quality 판단은 evaluator.py에서 수행한다.
  - 최종 리포트 생성은 report_builder.py에서 수행한다.
  - 이 파일은 질문 세트 단위의 점수 산정만 담당한다.
"""

from pydantic import BaseModel, Field

from interview.schemas.question import (
    QuestionKind,
    QuestionCategory,
    Difficulty,
)
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal
ACCURACY_BASE_WEIGHT = 0.7
SUFFICIENCY_BONUS_WEIGHT = 0.3

class AnswerAttempt(BaseModel):
    """질문 하나에 대한 답변 및 평가 기록.

    Attributes:
        answer_id:
            답변 ID.
            evaluator.py에서 AnswerQualitySignal을 생성할 때 만들어진다.

        question_id:
            사용자가 답변한 질문 ID.

        question_text:
            질문 원문.

        question_topic:
            질문 주제.
            예: FastAPI, JPA, JWT, Docker 등.

        question_kind:
            질문 종류.
            MAIN, FOLLOW_UP, CHALLENGE, CONFIRM_POSITIVE,
            CONFIRM_NEGATIVE, TRAP, HINT 등을 구분한다.

        question_category:
            질문 카테고리.
            TECHNICAL / PROJECT 등을 구분한다.

        question_difficulty:
            질문 난이도.
            EASY / MEDIUM / HARD 값을 가진다.

        answer_text:
            사용자의 답변 원문.

        signal:
            evaluator.py가 생성한 답변 평가 신호.
            quality, next_probe_target, rationale 등을 포함한다.

        delivery_metrics:
            음성 면접에서 사용하는 전달력 보조 지표.
            채팅 모드에서는 None일 수 있다.
    """
    answer_id: str

    question_id: str
    question_text: str
    question_topic: str
    question_kind: QuestionKind
    question_category: QuestionCategory
    question_difficulty: Difficulty

    answer_text: str
    signal: AnswerQualitySignal

    delivery_metrics: dict | None = None


class QuestionSetScore(BaseModel):
    """메인 질문과 파생 질문을 합산한 문항 최종 점수.

    Attributes:
        score:
            질문 세트 최종 점수.
            메인 질문 답변과 파생 질문 답변을 모두 반영한다.

        comment:
            질문 세트에 대한 평가 코멘트.
            FinalReport의 문항별 평가에 들어간다.
    """

    score: float = Field(ge=0.0, le=100.0)
    comment: str


def score_question_set(
    attempts: list[AnswerAttempt],
) -> QuestionSetScore:
    """질문 세트에 포함된 모든 답변을 이용해 최종 점수를 계산한다.

    Args:
        attempts:
            현재 질문 세트에 포함된 답변 시도 목록.
            일반적으로 메인 질문 답변 1개와 파생 질문 답변 0~3개가 들어간다.

    Returns:
        QuestionSetScore:
            FinalReport에 들어갈 문항 점수와 평가 코멘트.

    처리 흐름:
        1. 답변 기록이 없으면 0점 처리한다.
        2. 답변 기록이 있으면 임시 채점 함수로 점수를 산정한다.
        3. 추후 실제 채점 규칙 또는 LLM 기반 채점으로 교체할 수 있다.
    """

    return _calculate_question_set_score(attempts)




def _calculate_attempt_base_score(
    attempt: AnswerAttempt,
) -> float:
    accuracy = attempt.signal.accuracy
    sufficiency = attempt.signal.sufficiency

    effective_sufficiency = (
        accuracy * sufficiency
    )

    normalized_score = (
        accuracy * ACCURACY_BASE_WEIGHT
        + effective_sufficiency
        * SUFFICIENCY_BONUS_WEIGHT
    )

    score = normalized_score * 100

    return round(
        max(0.0, min(100.0, score)),
        2,
    )


def _calculate_question_set_score(
    attempts: list[AnswerAttempt],
) -> QuestionSetScore:
    """마지막 답변의 정확도와 충분성으로 질문 세트 기본 점수를 계산한다.

    accuracy를 기본 점수로 사용하고,
    sufficiency는 정확한 답변에 대한 추가 보상으로 반영한다.

    6-2에서 답변 이력에 따른 가산·감점을 추가하고,
    6-3에서 세트의 최종 quality 결정 규칙을 추가한다.
    """

    if not attempts:
        return QuestionSetScore(
            score=0.0,
            comment="평가할 답변이 없습니다.",
        )

    last_attempt = attempts[-1]
    base_score = _calculate_attempt_base_score(
        last_attempt
    )

    return QuestionSetScore(
        score=base_score,
        comment=(
            f"정확도 {last_attempt.signal.accuracy:.2f}, "
            f"충분성 {last_attempt.signal.sufficiency:.2f}를 "
            "반영해 기본 점수를 계산했습니다."
        ),
    )
    
def _find_main_attempt(attempts: list[AnswerAttempt]) -> AnswerAttempt:
    """질문 세트에서 메인 질문 답변을 찾는다."""

    for attempt in attempts:
        if attempt.question_kind == QuestionKind.MAIN:
            return attempt

    return attempts[0]