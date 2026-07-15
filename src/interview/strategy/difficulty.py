"""난이도 조정 로직.

직전 답변 평가와 지금까지의 난이도 분포를 보고 다음 질문 난이도를 정한다.
순수 함수로 두어 테스트하기 쉽게 한다 (LLM 호출 없음).
"""

from interview.schemas.question import Difficulty
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal
from interview.strategy.state import StrategyState

# 난이도 단계 순서. _step_up/_step_down에서 인덱스 이동으로 상승/하강을 표현
_DIFFICULTY_ORDER = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
# 연속 판단 기준 횟수. EASY 난이도 연속 출제 or SUFFICIENT 신호 연속 감지 시 상승.
_CONSECUTIVE_THRESHOLD = 2
# 해당 값 이상 메인 질문이 진행됐는데 HARD가 한 번도 안 나왔으면 강제로 HARD를 출제
_MIN_QUESTIONS_BEFORE_FORCE_HARD = 4

def _step_up(current: Difficulty) -> Difficulty:
    """난이도를 한 단계 올린다 (HARD면 유지)."""
    idx = _DIFFICULTY_ORDER.index(current)
    return _DIFFICULTY_ORDER[min(idx + 1, len(_DIFFICULTY_ORDER) - 1)]


def _step_down(current: Difficulty) -> Difficulty:
    """난이도를 한 단계 내린다 (EASY면 유지)."""
    idx = _DIFFICULTY_ORDER.index(current)
    return _DIFFICULTY_ORDER[max(idx - 1, 0)]


def _last_n_all_equal(items: list, n: int, value) -> bool:
    """최근 n개 항목이 모두 value와 같은지 확인한다. n개 미만이면 False."""
    recent = items[-n:]
    return len(recent) == n and all(item == value for item in recent)

def next_difficulty(
    state: StrategyState, 
    last_signal: AnswerQualitySignal | None
) -> Difficulty:
    """다음 메인 질문의 난이도를 결정.

    규칙(우선순위 순):
        1) 첫 질문(last_signal 없음) -> EASY
        2) 오개념/부정 확인(MISCONCEPTION, CONFIRM_NEGATIVE) -> 한 단계 하강
        3) EASY가 연속 _CONSECUTIVE_THRESHOLD회 -> 강제 상승 (쉬운 질문 편중 방지)
        4) 질문 수가 _MIN_QUESTIONS_BEFORE_FORCE_HARD 이상인데 HARD가
           한 번도 안 나왔으면 -> HARD로 강제 상승
        5) SUFFICIENT가 연속 _CONSECUTIVE_THRESHOLD회 -> 한 단계 상승
        6) 그 외 -> 직전 난이도 유지

    Args:
        state: 현재까지의 출제 이력.
        last_signal: 직전 답변 평가 결과. 첫 질문이면 None.

    Returns:
        다음 메인 질문에 적용할 Difficulty.
    """
    if last_signal is None:
        return Difficulty.EASY

    current = state.asked_difficulties[-1] if state.asked_difficulties else Difficulty.EASY

    if last_signal.quality in (AnswerQuality.MISCONCEPTION, AnswerQuality.CONFIRM_NEGATIVE):
        return _step_down(current)

    if _last_n_all_equal(state.asked_difficulties, _CONSECUTIVE_THRESHOLD, Difficulty.EASY):
        return _step_up(current)

    if (
        state.question_count >= _MIN_QUESTIONS_BEFORE_FORCE_HARD
        and Difficulty.HARD not in state.asked_difficulties
    ):
        return Difficulty.HARD

    if last_signal.quality == AnswerQuality.SUFFICIENT:
        if _last_n_all_equal(state.recent_qualities, _CONSECUTIVE_THRESHOLD, AnswerQuality.SUFFICIENT):
            return _step_up(current)
        return current

    return current
