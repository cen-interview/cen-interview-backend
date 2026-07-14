"""질문 세트 점수 산정에 사용하는 정책값."""

from interview.schemas.question import QuestionKind,Difficulty
from interview.schemas.signals import AnswerQuality


# accuracy를 기본 점수로 사용하고,
# sufficiency는 정확한 답변에 대한 추가 보상으로 반영한다.
ACCURACY_BASE_WEIGHT = 0.7
SUFFICIENCY_BONUS_WEIGHT = 0.3

# 채팅: content_score 100%
# 음성: content_score 90% + delivery_score 10%
CONTENT_SCORE_WEIGHT = 0.9
DELIVERY_SCORE_WEIGHT = 0.1


# 파생 질문에서 문제를 해결했을 때 적용하는 보정 비율.
RESOLVED_RATE_BY_KIND: dict[QuestionKind, float] = {
    QuestionKind.FOLLOW_UP: 0.05,
    QuestionKind.CHALLENGE: -0.10,
    QuestionKind.CONFIRM_POSITIVE: 0.02,
    QuestionKind.CONFIRM_NEGATIVE: -0.03,
    QuestionKind.TRAP: 0.05,
    QuestionKind.HINT: -0.10,
}


# 파생 질문에서도 문제를 해결하지 못했을 때 적용하는 보정 비율.
UNRESOLVED_RATE_BY_KIND: dict[QuestionKind, float] = {
    QuestionKind.FOLLOW_UP: -0.05,
    QuestionKind.CHALLENGE: -0.30,
    QuestionKind.CONFIRM_POSITIVE: -0.05,
    QuestionKind.CONFIRM_NEGATIVE: -0.15,
    QuestionKind.TRAP: -0.20,
    QuestionKind.HINT: -0.25,
}


# 각 파생 질문이 어떤 이전 quality를 검증하기 위해 생성됐는지 나타낸다.
EXPECTED_PRIOR_QUALITY_BY_KIND: dict[
    QuestionKind,
    AnswerQuality,
] = {
    QuestionKind.FOLLOW_UP:
        AnswerQuality.BONUS_AVAILABLE,

    QuestionKind.CHALLENGE:
        AnswerQuality.MISCONCEPTION,

    QuestionKind.CONFIRM_POSITIVE:
        AnswerQuality.CONFIRM_POSITIVE,

    QuestionKind.CONFIRM_NEGATIVE:
        AnswerQuality.CONFIRM_NEGATIVE,

    QuestionKind.TRAP:
        AnswerQuality.TRAP_AVAILABLE,
}


DIFFICULTY_MULTIPLIER = {
    Difficulty.EASY: 0.97,
    Difficulty.MEDIUM: 1.00,
    Difficulty.HARD: 1.03,
}