"""평가 신호를 면접관의 간접적인 반응 정책으로 변환한다."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


class ReactionTone(str, Enum):
    """직전 답변에 대한 면접관 리액션의 강도."""

    NEUTRAL = "neutral"
    POSITIVE = "positive"
    CAUTIOUS = "cautious"
    CORRECTIVE = "corrective"


@dataclass(frozen=True)
class ReactionPolicy:
    """발화 생성기가 사용할 평가 기반 반응 정책.

    Attributes:
        quality:
            직전 답변의 평가 품질. 평가 신호가 없거나 올바르지 않으면 None이다.

        tone:
            발화에서 드러낼 반응 강도. 점수나 정답을 직접 공개하지 않고
            긍정, 유보, 재검토 필요 여부만 전달한다.

        guidance:
            LLM이 리액션을 만들 때 따를 안전한 표현 지침. 평가의 상세 근거나
            정답 대신 설명 범위, 개념 구분, 논리, 일관성 같은 관점만 제공한다.
    """

    quality: AnswerQuality | None
    tone: ReactionTone
    guidance: str


_REACTION_POLICIES = {
    AnswerQuality.SUFFICIENT: ReactionPolicy(
        quality=AnswerQuality.SUFFICIENT,
        tone=ReactionTone.POSITIVE,
        guidance=(
            "답변이 충분했다는 점을 절제된 긍정 표현으로 알려 주세요. "
            "과장된 칭찬이나 점수 표현은 사용하지 마세요."
        ),
    ),
    AnswerQuality.UNKNOWN: ReactionPolicy(
        quality=AnswerQuality.UNKNOWN,
        tone=ReactionTone.NEUTRAL,
        guidance=(
            "사용자가 답변하기 어려워하는 상황이다. "
            "평가 결과나 부족함을 직접 드러내지 말고 "
            "짧고 중립적으로 다음 질문으로 넘어가세요."
        ),
    ),
    AnswerQuality.BONUS_AVAILABLE: ReactionPolicy(
        quality=AnswerQuality.BONUS_AVAILABLE,
        tone=ReactionTone.POSITIVE,
        guidance=(
            "답변의 기본 방향이 좋았음을 인정하되, 설명을 더 확장할 여지가 "
            "있다는 인상만 남겨 주세요."
        ),
    ),
    AnswerQuality.CONFIRM_POSITIVE: ReactionPolicy(
        quality=AnswerQuality.CONFIRM_POSITIVE,
        tone=ReactionTone.CAUTIOUS,
        guidance=(
            "전반적인 방향은 타당하지만 설명의 범위나 조건이 아직 분명하지 "
            "않다는 점을 정중하게 암시해 주세요."
        ),
    ),
    AnswerQuality.TRAP_AVAILABLE: ReactionPolicy(
        quality=AnswerQuality.TRAP_AVAILABLE,
        tone=ReactionTone.CAUTIOUS,
        guidance=(
            "유사한 개념을 구분하는 기준이 중요하다는 점을 암시하되, 어떤 "
            "개념을 구분해야 하는지는 직접 말하지 마세요."
        ),
    ),
    AnswerQuality.MISCONCEPTION: ReactionPolicy(
        quality=AnswerQuality.MISCONCEPTION,
        tone=ReactionTone.CORRECTIVE,
        guidance=(
            "답변의 전제나 개념 사이 관계를 다시 짚을 필요가 있음을 정중하지만 "
            "분명하게 드러내세요. 틀렸다고 단정하거나 정답을 알려주지는 마세요."
        ),
    ),
    AnswerQuality.CONFIRM_NEGATIVE: ReactionPolicy(
        quality=AnswerQuality.CONFIRM_NEGATIVE,
        tone=ReactionTone.CORRECTIVE,
        guidance=(
            "근거나 이전 설명과의 일관성을 다시 살펴볼 필요가 있음을 정중하지만 "
            "분명하게 드러내세요. 충돌 내용을 직접 공개하지는 마세요."
        ),
    ),
}

_NEUTRAL_POLICY = ReactionPolicy(
    quality=None,
    tone=ReactionTone.NEUTRAL,
    guidance="평가 신호가 없으므로 현재 상황의 기능만 짧고 중립적으로 수행하세요.",
)


def select_reaction_policy(
    raw_signal: AnswerQualitySignal | dict[str, Any] | None,
) -> ReactionPolicy:
    """직전 평가 신호를 면접관 발화용 반응 정책으로 변환한다.

    LangGraph 상태에는 평가 신호가 Pydantic 모델 또는 JSON 호환 dict 형태로
    저장될 수 있다. 발화 레이어는 전체 평가 근거를 직접 노출하지 않고 quality만
    읽어 미리 정의한 반응 강도와 표현 지침을 선택한다. 신호가 없거나 알 수 없는
    quality이면 세션 흐름을 막지 않고 중립 정책을 반환한다.

    Args:
        raw_signal:
            직전 답변의 평가 신호. AnswerQualitySignal, 같은 필드를 가진 dict,
            또는 평가가 없는 경우 None이다.

    Returns:
        평가 결과에 대응하는 ReactionPolicy. 유효한 평가 품질을 확인할 수
        없으면 중립 정책을 반환한다.
    """
    if raw_signal is None:
        return _NEUTRAL_POLICY

    raw_quality = (
        raw_signal.quality
        if isinstance(raw_signal, AnswerQualitySignal)
        else raw_signal.get("quality")
    )
    try:
        quality = (
            raw_quality
            if isinstance(raw_quality, AnswerQuality)
            else AnswerQuality(raw_quality)
        )
    except (TypeError, ValueError):
        return _NEUTRAL_POLICY

    return _REACTION_POLICIES.get(quality, _NEUTRAL_POLICY)
