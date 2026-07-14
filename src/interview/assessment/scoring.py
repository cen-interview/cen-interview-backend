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
from interview.assessment.scoring_policy import (
    ACCURACY_BASE_WEIGHT,
    SUFFICIENCY_BONUS_WEIGHT,
    RESOLVED_RATE_BY_KIND,
    UNRESOLVED_RATE_BY_KIND,
    EXPECTED_PRIOR_QUALITY_BY_KIND,
    DIFFICULTY_MULTIPLIER,
    CONTENT_SCORE_WEIGHT,
    DELIVERY_SCORE_WEIGHT,
)


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


# 질문 세트의 답변 이력을 반영해 문항 최종 점수를 계산한다.
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

# 점수를 0~100 범위로 제한하고 소수점 둘째 자리까지 반올림한다.
def _clamp_score(score: float) -> float:

    return round(
        max(0.0, min(100.0, score)),
        2,
    )

# 기본 점수에 비율 방식의 가산점 또는 감점을 적용한다.
def _apply_score_rate(
    score: float,
    rate: float,
) -> float:

    adjusted_score = score * (1.0 + rate)

    return _clamp_score(adjusted_score)

# 메인 질문의 난이도 배율을 질문 세트 점수에 적용한다.
def _apply_difficulty_adjustment(
    score: float,
    attempts: list[AnswerAttempt],
) -> tuple[float, float]:

    main_attempt = _find_main_attempt(attempts)

    multiplier = DIFFICULTY_MULTIPLIER.get(
        main_attempt.question_difficulty,
        1.0,
    )

    adjusted_score = _clamp_score(
        score * multiplier
    )

    return adjusted_score, multiplier

# 답변의 정확도와 충분성을 이용해 내용 기본 점수를 계산한다.
def _calculate_attempt_base_score(
    attempt: AnswerAttempt,
) -> float:
    """답변 하나의 accuracy와 sufficiency로 기본 점수를 계산한다.

    accuracy를 전체 점수의 기준으로 사용하고,
    sufficiency는 정확한 답변에 대한 추가 보상으로 반영한다.

    따라서 accuracy가 0이면 sufficiency가 높아도 점수는 0이다.
    """
    
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

    return _clamp_score(
        normalized_score * 100
    )

# 마지막 답변 이전에 지정된 quality 판정이 있었는지 확인한다.
def _has_prior_quality(
    attempts: list[AnswerAttempt],
    quality: AnswerQuality,
) -> bool:

    return any(
        attempt.signal.quality == quality
        for attempt in attempts[:-1]
    )

# 파생 질문의 해결 여부에 맞는 질문 세트 평가 문장을 반환한다
def _build_adjustment_comment(
    question_kind: QuestionKind,
    resolved: bool,
) -> str:
    """파생 질문의 해결 여부에 맞는 평가 문구를 반환한다."""

    comments = {
        QuestionKind.FOLLOW_UP: {
            True: (
                "누락된 내용을 꼬리질문에서 "
                "보완했습니다."
            ),
            False: (
                "꼬리질문 이후에도 설명이 "
                "충분히 보완되지 않았습니다."
            ),
        },
        QuestionKind.CHALLENGE: {
            True: (
                "초기 오개념을 압박 질문 이후 "
                "정정했습니다."
            ),
            False: (
                "압박 질문 이후에도 오개념이 "
                "해소되지 않았습니다."
            ),
        },
        QuestionKind.CONFIRM_POSITIVE: {
            True: (
                "적용 범위와 세부 사실을 "
                "명확히 확인했습니다."
            ),
            False: (
                "적용 범위와 세부 사실이 "
                "충분히 확인되지 않았습니다."
            ),
        },
        QuestionKind.CONFIRM_NEGATIVE: {
            True: (
                "근거와의 불일치를 인정하고 "
                "답변을 정정했습니다."
            ),
            False: (
                "근거와의 불일치가 "
                "해소되지 않았습니다."
            ),
        },
        QuestionKind.TRAP: {
            True: (
                "유사 개념과 경계 조건을 "
                "정확히 구분했습니다."
            ),
            False: (
                "유사 개념과 경계 조건을 "
                "정확히 구분하지 못했습니다."
            ),
        },
    }

    return comments[question_kind][resolved]

# 파생 질문에서 이전 부족점이나 오개념을 해결했는지 점수에 반영한다.
def _apply_history_adjustment(
    base_score: float,
    attempts: list[AnswerAttempt],
) -> tuple[float, str]:
    """흐름:
    - FOLLOW_UP: 설명 누락 보완 여부
    - CHALLENGE: 오개념 정정 여부
    - CONFIRM_POSITIVE: 세부 사실 확인 여부
    - CONFIRM_NEGATIVE: 충돌 정정 여부
    - TRAP: 유사 개념 구분 여부
    - HINT: 힌트 이후 답변 회복 여부
    """

    if len(attempts) < 2:
        return (
            _clamp_score(base_score),
            "첫 답변의 정확도와 충분성을 반영했습니다.",
        )

    last_attempt = attempts[-1]
    last_kind = last_attempt.question_kind
    last_quality = last_attempt.signal.quality

    resolved = (
        last_quality == AnswerQuality.SUFFICIENT
    )

    # HINT는 특정 quality 신호로 생성되는 질문이 아니므로
    # 이전 quality를 확인하지 않고 힌트 사용 자체를 보정한다.
    if last_kind == QuestionKind.HINT:
        rate = (
            RESOLVED_RATE_BY_KIND[last_kind]
            if resolved
            else UNRESOLVED_RATE_BY_KIND[last_kind]
        )

        comment = (
            "힌트 이후 핵심 답변을 회복했습니다."
            if resolved
            else (
                "힌트 이후에도 핵심 답변을 "
                "충분히 회복하지 못했습니다."
            )
        )

        return (
            _apply_score_rate(base_score, rate),
            comment,
        )

    expected_prior_quality = (
        EXPECTED_PRIOR_QUALITY_BY_KIND.get(
            last_kind
        )
    )

    # MAIN이거나 이전 quality와 대응되지 않는 파생 질문이면
    # 이력 보정을 적용하지 않는다.
    if (
        expected_prior_quality is None
        or not _has_prior_quality(
            attempts,
            expected_prior_quality,
        )
    ):
        return (
            _clamp_score(base_score),
            "마지막 답변의 정확도와 충분성을 반영했습니다.",
        )

    rate = (
        RESOLVED_RATE_BY_KIND[last_kind]
        if resolved
        else UNRESOLVED_RATE_BY_KIND[last_kind]
    )

    adjusted_score = _apply_score_rate(
        base_score,
        rate,
    )

    comment = _build_adjustment_comment(
        question_kind=last_kind,
        resolved=resolved,
    )

    return adjusted_score, comment

# 내용, 답변 이력, 난이도와 전달력을 반영해 질문 세트 최종 점수를 계산한다.
def _calculate_question_set_score(
    attempts: list[AnswerAttempt],
) -> QuestionSetScore:

    if not attempts:
        return QuestionSetScore(
            score=0.0,
            comment="평가할 답변이 없습니다.",
        )

    last_attempt = attempts[-1]
    base_score = _calculate_attempt_base_score(
        last_attempt
    )

    history_adjusted_score, adjustment_comment = (
        _apply_history_adjustment(
            base_score=base_score,
            attempts=attempts,
        )   
    )
    difficulty_adjusted_score, _ = (
        _apply_difficulty_adjustment(
            score=history_adjusted_score,
            attempts=attempts,
        )
    )

    content_score = difficulty_adjusted_score

    delivery_score = _calculate_set_delivery_score(
        attempts
    )

    final_score = (
        _combine_content_and_delivery_score(
            content_score=content_score,
            delivery_score=delivery_score,
        )
    )

    return QuestionSetScore(
        score=final_score,
        comment=adjustment_comment,
    )

# 질문 세트에서 기준이 되는 메인 질문 답변을 찾는다.    
def _find_main_attempt(
    attempts: list[AnswerAttempt],
    ) -> AnswerAttempt:

    for attempt in attempts:
        if attempt.question_kind == QuestionKind.MAIN:
            return attempt

    return attempts[0]

# ---------------------------------------------- 

# 분당 발화 속도를 0~100 범위의 전달력 점수로 변환한다.
def _calculate_speech_rate_score(
    speech_rate_wpm: float,
) -> float:
    if 120 <= speech_rate_wpm <= 170:
        return 100.0

    if 100 <= speech_rate_wpm < 120:
        return 85.0

    if 170 < speech_rate_wpm <= 200:
        return 80.0

    return 60.0

# 필러 표현 사용 횟수를 0~100 범위의 전달력 점수로 변환한다.
def _calculate_filler_score(
    filler_count: int,
) -> float:

    if filler_count <= 2:
        return 100.0

    if filler_count <= 5:
        return 85.0

    if filler_count <= 9:
        return 70.0

    return 50.0

# 답변 하나의 음성 지표를 가중 평균한 전달력 점수로 변환한다.
def _calculate_attempt_delivery_score(
    delivery_metrics: dict | None,
) -> float | None:

    # metrics 자체가 없으면 전달력 계산을 하지 않는다.
    if not delivery_metrics:
        return None

    speech_rate_wpm = delivery_metrics.get(
        "speech_rate_wpm"
    )
    filler_count = delivery_metrics.get(
        "filler_count"
    )

    scores: list[tuple[float, float]] = []

    if speech_rate_wpm is not None:
        scores.append(
            (
                _calculate_speech_rate_score(
                    speech_rate_wpm
                ),
                0.6,
            )
        )

    if filler_count is not None:
        scores.append(
            (
                _calculate_filler_score(
                    filler_count
                ),
                0.4,
            )
        )

    if not scores:
        return None

    total_weight = sum(
        weight
        for _, weight in scores
    )

    weighted_score = sum(
        score * weight
        for score, weight in scores
    ) / total_weight

    return _clamp_score(weighted_score)

# 질문 세트에 포함된 음성 답변들의 전달력 점수 평균을 계산한다.
def _calculate_set_delivery_score(
    attempts: list[AnswerAttempt],
) -> float | None:

    delivery_scores = []

    for attempt in attempts:
        attempt_delivery_score = (
            _calculate_attempt_delivery_score(
                attempt.delivery_metrics
            )
        )

        # metrics가 없는 답변은 세트 평균에서 제외한다.
        if attempt_delivery_score is not None:
            delivery_scores.append(
                attempt_delivery_score
            )

  

    if not delivery_scores:
        return None

    return _clamp_score(
        sum(delivery_scores)
        / len(delivery_scores)
    )

# 음성 지표가 있으면 내용 점수와 전달력 점수를 가중 결합한다.
def _combine_content_and_delivery_score(
    content_score: float,
    delivery_score: float | None,
) -> float:

    if delivery_score is None:
        return _clamp_score(content_score)

    final_score = (
        content_score * CONTENT_SCORE_WEIGHT
        + delivery_score * DELIVERY_SCORE_WEIGHT
    )

    return _clamp_score(final_score)