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


def _calculate_question_set_score(
    attempts: list[AnswerAttempt],
) -> QuestionSetScore:
    """질문 세트의 최종 점수를 계산한다.

    Args:
        attempts:
            현재 질문 세트에 포함된 답변 시도 목록.
            메인 질문 답변 1개와 파생 질문 답변 0~3개가 포함될 수 있다.

    Returns:
        QuestionSetScore:
            문항 최종 점수와 평가 코멘트.

    점수 산정 방향:
        - 마지막 답변의 quality를 기준으로 기본 점수를 정한다.
        - 질문 난이도(question_difficulty)에 따라 점수를 보정할 수 있다.
        - 질문 카테고리(question_category)에 따라 평가 기준을 다르게 적용할 수 있다.
        - 파생 질문에서 답변이 보완되었는지도 반영한다.


    임시 채점 기준:
        - 첫 답변에서 충분히 답변한 경우: 90점
        - 파생 질문을 통해 보완한 경우: 80점
        - 긍정 확인이 필요한 경우: 70점
        - 추가 설명이 필요한 경우: 65점
        - 함정 질문이 필요한 경우: 60점
        - 근거 또는 이전 답변과 충돌하는 경우: 50점
        - 오개념이 남아 있는 경우: 40점

    TODO:
        추후 아래 요소를 반영하여 정교한 점수 산정으로 개선한다.
        - Evidence와의 일치도
        - 핵심 개념 포함 여부
        - 파생 질문에서 보완된 정도
        - 오개념 지속 여부
        - 프로젝트 적용 사례의 구체성
        - 음성 모드의 전달력 지표
    """

    main_attempt = _find_main_attempt(attempts)
    last_attempt = attempts[-1]

    final_quality = last_attempt.signal.quality
    difficulty = main_attempt.question_difficulty
    category = main_attempt.question_category
    derived_count = len(attempts) - 1

    # 첫 답변에서 충분했던 경우
    if (
        len(attempts) == 1
        and final_quality == AnswerQuality.SUFFICIENT
    ):
        return QuestionSetScore(
            score=90.0,
            comment="첫 답변에서 핵심 내용을 충분히 설명했습니다.",
        )

    # 첫 답변은 부족했지만 파생 질문에서 보완한 경우
    if (
        len(attempts) > 1
        and final_quality == AnswerQuality.SUFFICIENT
    ):
        return QuestionSetScore(
            score=80.0,
            comment="초기 답변의 부족한 부분을 파생 질문에서 보완했습니다.",
        )

    if final_quality == AnswerQuality.MISCONCEPTION:
        return QuestionSetScore(
            score=40.0,
            comment="답변에 오개념 또는 논리적 문제가 남아 있습니다.",
        )

    if final_quality == AnswerQuality.CONFIRM_NEGATIVE:
        return QuestionSetScore(
            score=50.0,
            comment="근거 또는 이전 답변과 충돌하는 내용이 남아 있습니다.",
        )

    if final_quality == AnswerQuality.TRAP_AVAILABLE:
        return QuestionSetScore(
            score=60.0,
            comment="유사한 개념을 구분하는 추가 확인이 필요합니다.",
        )

    if final_quality == AnswerQuality.CONFIRM_POSITIVE:
        return QuestionSetScore(
            score=70.0,
            comment="답변은 대체로 맞지만 적용 범위나 사실관계에 대한 추가 확인이 필요합니다.",
        )

    return QuestionSetScore(
        score=65.0,
        comment="답변은 대체로 맞지만 추가 설명이 필요합니다.",
    )
    
def _find_main_attempt(attempts: list[AnswerAttempt]) -> AnswerAttempt:
    """질문 세트에서 메인 질문 답변을 찾는다."""

    for attempt in attempts:
        if attempt.question_kind == QuestionKind.MAIN:
            return attempt

    return attempts[0]