"""next_difficulty() 난이도 결정 로직 단위 테스트.

StrategyState/AnswerQualitySignal 타입만 사용하는 순수 로직 테스트이므로
다른 에이전트(Evidence/Assessment) 호출이나 실제 LLM 호출이 없다.
"""

import pytest

from interview.schemas.question import Difficulty
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal
from interview.strategy.difficulty import next_difficulty
from interview.strategy.state import StrategyState


def _signal(quality: AnswerQuality) -> AnswerQualitySignal:
    return AnswerQualitySignal(answer_id="a-1", question_id="q-1", quality=quality)


def test_next_difficulty_first_question_is_easy():
    """규칙 1: 첫 질문(last_signal=None) -> EASY"""
    state = StrategyState()
    assert next_difficulty(state, None) == Difficulty.EASY


@pytest.mark.parametrize(
    "current_difficulty,quality,expected",
    [
        (Difficulty.MEDIUM, AnswerQuality.MISCONCEPTION, Difficulty.EASY),
        (Difficulty.HARD, AnswerQuality.MISCONCEPTION, Difficulty.MEDIUM),
        (Difficulty.MEDIUM, AnswerQuality.CONFIRM_NEGATIVE, Difficulty.EASY),
        (Difficulty.EASY, AnswerQuality.MISCONCEPTION, Difficulty.EASY),  # 이미 최저, 유지
        (Difficulty.HARD, AnswerQuality.UNKNOWN, Difficulty.MEDIUM),  # "모르겠다"도 즉시 하강
        (Difficulty.MEDIUM, AnswerQuality.UNKNOWN, Difficulty.EASY),
        (Difficulty.EASY, AnswerQuality.UNKNOWN, Difficulty.EASY),  # 이미 최저, 유지
    ],
)
def test_next_difficulty_steps_down_on_negative_signal(current_difficulty, quality, expected):
    """규칙 2: MISCONCEPTION/CONFIRM_NEGATIVE/UNKNOWN -> 한 단계 하강

    UNKNOWN("모르겠다")도 오개념과 동일하게 즉시 하강시킨다 - 그렇지 않으면
    지원자가 계속 모른다고 답해도 난이도가 HARD에 고정된 채 내려오지 않는다.
    """
    state = StrategyState(asked_difficulties=[current_difficulty])
    assert next_difficulty(state, _signal(quality)) == expected


def test_next_difficulty_unknown_wins_over_forced_up_on_easy_streak():
    """규칙 2가 규칙 3(연속 EASY 강제 상승)보다 우선한다.

    EASY가 2연속이면 원래는 규칙 3이 강제로 MEDIUM까지 올리지만(다른 테스트
    test_next_difficulty_forces_up_after_two_consecutive_easy 참고), 이번엔
    "모르겠다"는 신호가 왔으므로 억지로 올리지 않고 하강(EASY 유지)이 이긴다.
    """
    state = StrategyState(asked_difficulties=[Difficulty.EASY, Difficulty.EASY])
    result = next_difficulty(state, _signal(AnswerQuality.UNKNOWN))
    assert result == Difficulty.EASY


def test_next_difficulty_forces_up_after_two_consecutive_easy():
    """규칙 3: EASY 2연속 -> 강제 상승"""
    state = StrategyState(asked_difficulties=[Difficulty.EASY, Difficulty.EASY])
    result = next_difficulty(state, _signal(AnswerQuality.BONUS_AVAILABLE))
    assert result == Difficulty.MEDIUM


def test_next_difficulty_does_not_force_up_with_single_easy():
    """규칙 3 반례: EASY가 1개뿐이면 강제 상승 안 함"""
    state = StrategyState(asked_difficulties=[Difficulty.MEDIUM, Difficulty.EASY])
    result = next_difficulty(state, _signal(AnswerQuality.BONUS_AVAILABLE))
    assert result == Difficulty.EASY  # 유지 (현재값 그대로)


def test_next_difficulty_forces_hard_when_none_appeared():
    """규칙 4: 질문 5개 이상인데 HARD 미출제 -> HARD 강제"""
    state = StrategyState(
        asked_difficulties=[Difficulty.MEDIUM, Difficulty.EASY, Difficulty.MEDIUM, Difficulty.EASY],
        question_count=5,
    )
    result = next_difficulty(state, _signal(AnswerQuality.BONUS_AVAILABLE))
    assert result == Difficulty.HARD


def test_next_difficulty_does_not_force_hard_before_threshold():
    """규칙 4 반례: 질문 수가 threshold 미만이면 강제 안 함"""
    state = StrategyState(
        asked_difficulties=[Difficulty.MEDIUM],
        question_count=2,
    )
    result = next_difficulty(state, _signal(AnswerQuality.BONUS_AVAILABLE))
    assert result == Difficulty.MEDIUM  # 유지


def test_next_difficulty_steps_up_after_two_consecutive_sufficient():
    """규칙 5: SUFFICIENT 2연속 -> 한 단계 상승"""
    state = StrategyState(
        asked_difficulties=[Difficulty.EASY, Difficulty.MEDIUM],  # EASY 2연속 아님
        recent_qualities=[AnswerQuality.SUFFICIENT, AnswerQuality.SUFFICIENT],
    )
    result = next_difficulty(state, _signal(AnswerQuality.SUFFICIENT))
    assert result == Difficulty.HARD


def test_next_difficulty_does_not_step_up_with_single_sufficient():
    """규칙 5 반례: SUFFICIENT 1회만으로는 상승 안 함"""
    state = StrategyState(
        asked_difficulties=[Difficulty.EASY, Difficulty.MEDIUM],
        recent_qualities=[AnswerQuality.BONUS_AVAILABLE, AnswerQuality.SUFFICIENT],
    )
    result = next_difficulty(state, _signal(AnswerQuality.SUFFICIENT))