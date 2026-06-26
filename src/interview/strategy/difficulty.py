"""난이도 조정 로직.

직전 답변 평가와 지금까지의 난이도 분포를 보고 다음 질문 난이도를 정한다.
순수 함수로 두어 테스트하기 쉽게 한다 (LLM 호출 없음).
"""

from interview.schemas.question import Difficulty
from interview.schemas.signals import AnswerQualitySignal, QualityLevel
from interview.strategy.state import StrategyState


def next_difficulty(
    state: StrategyState, last_signal: AnswerQualitySignal | None
) -> Difficulty:
    """다음 질문 난이도 결정.

    규칙 예시 (TODO 담당 B: 실제 규칙은 팀에서 합의):
      - 직전 답변이 충분 → 한 단계 올림
      - 얕음/막힘     → 유지하거나 내림
      - 쉬운 질문만 계속 나오지 않게 균형도 고려
    """
    if last_signal is None:
        return Difficulty.EASY
    if last_signal.quality == QualityLevel.SUFFICIENT:
        return Difficulty.MEDIUM  # TODO: 단계적 상승 로직
    return Difficulty.EASY
