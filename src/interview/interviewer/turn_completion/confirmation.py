"""종료 확인 이후의 지원자 응답 의도를 분류한다."""

import asyncio
import re
from typing import Any

from interview.config import settings
from interview.interviewer.intent import normalize_utterance
from interview.interviewer.turn_completion.models import ConfirmationIntentDecision
from interview.interviewer.turn_completion.prompts import (
    CONFIRMATION_INTENT_SYSTEM_PROMPT,
    build_confirmation_intent_user_prompt,
)
from interview.llm.client import get_llm


_FINISH_PATTERNS: tuple[str, ...] = (
    r"(?:네|예|맞습니다|그렇습니다)",
    r"(?:답변은\s*)?(?:여기까지|끝|이상)(?:입니다|이에요|예요|요)?",
    r"(?:끝났습니다|마쳤습니다)",
)

_CONTINUE_PATTERNS: tuple[str, ...] = (
    r"(?:아니요|아니오|아닙니다)",
    r"(?:잠시만|잠시만요|잠깐만|잠깐만요)",
    r"(?:아직|아직이요|아직입니다)",
    r"(?:조금만\s*)?더\s*(?:말씀드리겠습니다|말하겠습니다|말할게요)",
    r"(?:조금만\s*)?더\s*생각(?:해\s*볼게요|하겠습니다|할게요)",
)

_CONTROL_PREFIX_PATTERN = re.compile(
    r"^\s*(?:네|예|맞습니다|아니요|아니오|아닙니다|"
    r"잠시만|잠시만요|잠깐만|잠깐만요)"
    r"(?:\s+|[,，.!?！？:;]\s*)(?P<content>.+?)\s*$",
)

_NON_SUBSTANTIVE_CONTENT = {
    "그리고",
    "추가로",
    "그런데",
    "또",
    "더",
    "음",
    "어",
    "그",
}


class ConfirmationIntentClassifier:
    """답변 종료 확인에 대한 지원자의 다음 발화를 분류한다.

    명확한 종료와 계속 표현은 규칙으로 먼저 처리한다. 규칙으로 구분되지 않는
    응답만 낮은 temperature의 구조화 LLM으로 분류하며, 실패하면 자동 제출을
    막기 위해 ``unknown``을 반환한다.

    Attributes:
        _llm:
            ConfirmationIntentDecision 구조화 출력을 지원하는 LLM client.

        _timeout_seconds:
            애매한 확인 응답의 LLM 분류를 기다릴 최대 시간.
    """

    def __init__(
        self,
        *,
        llm: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """확인 응답 의도 판정기를 생성한다.

        Args:
            llm:
                선택적으로 주입할 LLM client. 없으면 공용 LLM을 temperature
                0으로 생성해 사용한다.

            timeout_seconds:
                LLM 분류 응답을 기다릴 최대 시간. 없으면 완료 판단과 같은
                애플리케이션 timeout 설정을 사용한다.

        Raises:
            ValueError:
                timeout_seconds가 0 이하인 경우.
        """
        configured_timeout = (
            settings.turn_completion_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if configured_timeout <= 0:
            raise ValueError("확인 응답 분류 제한 시간은 0보다 커야 합니다.")

        self._llm = llm if llm is not None else get_llm(temperature=0.0)
        self._timeout_seconds = configured_timeout

    async def classify(self, response_text: str) -> ConfirmationIntentDecision:
        """종료 확인 이후의 지원자 응답 의도를 분류한다.

        명확한 제어 표현과 제어 접두사가 붙은 추가 설명은 규칙으로 처리한다.
        나머지 응답은 구조화 LLM으로 분류하되, answer_content에 사용할 문자열은
        LLM이 생성한 문장이 아니라 원본 응답에서 결정한다.

        Args:
            response_text:
                종료 확인 질문 이후 STT가 만든 지원자 응답 원문.

        Returns:
            finish, continue, answer_content 또는 unknown 의도와 선택적인 추가
            답변 내용을 담은 판단 결과.
        """
        rule_decision = _classify_with_rules(response_text)
        if rule_decision is not None:
            return rule_decision

        if not normalize_utterance(response_text):
            return _unknown_fallback()

        messages = [
            {"role": "system", "content": CONFIRMATION_INTENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_confirmation_intent_user_prompt(response_text),
            },
        ]

        try:
            structured_llm = self._llm.with_structured_output(
                ConfirmationIntentDecision
            )
            raw_decision = await asyncio.wait_for(
                structured_llm.ainvoke(messages),
                timeout=self._timeout_seconds,
            )
            decision = (
                raw_decision
                if isinstance(raw_decision, ConfirmationIntentDecision)
                else ConfirmationIntentDecision.model_validate(raw_decision)
            )
            return _replace_with_original_answer_content(
                response_text=response_text,
                decision=decision,
            )
        except Exception:
            return _unknown_fallback()


def _classify_with_rules(
    response_text: str,
) -> ConfirmationIntentDecision | None:
    """명확한 확인 응답을 문자열 규칙으로 우선 분류한다.

    Args:
        response_text:
            종료 확인 이후의 지원자 응답 원문.

    Returns:
        규칙으로 의도가 명확하면 해당 판단 결과. 애매해 LLM 분류가 필요하면
        None.
    """
    normalized = normalize_utterance(response_text)
    if not normalized:
        return _unknown_fallback()

    if _fullmatches_any(normalized, _FINISH_PATTERNS):
        return ConfirmationIntentDecision(
            intent="finish",
            confidence=1.0,
        )
    if _fullmatches_any(normalized, _CONTINUE_PATTERNS):
        return ConfirmationIntentDecision(
            intent="continue",
            confidence=1.0,
        )

    answer_content = _extract_answer_content(response_text)
    if answer_content is not None and _has_substantive_content(answer_content):
        return ConfirmationIntentDecision(
            intent="answer_content",
            answer_content=answer_content,
            confidence=1.0,
        )
    return None


def _fullmatches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """정규화된 전체 발화와 일치하는 패턴이 있는지 확인한다.

    Args:
        text:
            intent.py의 normalize_utterance로 정규화한 발화.

        patterns:
            전체 일치로 비교할 정규 표현식 패턴 목록.

    Returns:
        하나 이상의 패턴이 발화 전체와 일치하면 True.
    """
    return any(re.fullmatch(pattern, text) is not None for pattern in patterns)


def _extract_answer_content(response_text: str) -> str | None:
    """확인 제어 접두사를 제외한 실질적인 응답 원문을 추출한다.

    ``네, 그리고...`` 또는 ``아니요, 추가로...``처럼 제어 응답과 추가
    설명이 섞였을 때 앞의 제어 표현만 제거하고 나머지 문장은 유지한다.

    Args:
        response_text:
            종료 확인 이후의 지원자 응답 원문.

    Returns:
        제어 접두사 뒤의 원문 내용. 접두사가 없거나 뒤 내용이 없으면 None.
    """
    matched = _CONTROL_PREFIX_PATTERN.fullmatch(response_text)
    if matched is None:
        return None
    content = matched.group("content").strip()
    return content or None


def _has_substantive_content(text: str) -> bool:
    """추출한 문자열에 실제 답변으로 연결할 내용이 있는지 확인한다.

    Args:
        text:
            제어 접두사를 제거한 응답 문자열.

    Returns:
        연결 표현이나 한두 음절의 머뭇거림이 아닌 내용이 있으면 True.
    """
    normalized = normalize_utterance(text)
    if not normalized or normalized in _NON_SUBSTANTIVE_CONTENT:
        return False
    return len(normalized.replace(" ", "")) >= 2


def _replace_with_original_answer_content(
    *,
    response_text: str,
    decision: ConfirmationIntentDecision,
) -> ConfirmationIntentDecision:
    """LLM의 추가 답변 분류에 원문 기반 내용을 적용한다.

    LLM은 의도만 결정하도록 신뢰하고, 실제 답변에 이어 붙일 문자열은 원본
    전사문에서 가져온다. 제어 접두사가 없으면 원문 전체를 사용한다.

    Args:
        response_text:
            종료 확인 이후의 지원자 응답 원문.

        decision:
            구조화 LLM이 반환한 확인 응답 의도.

    Returns:
        answer_content 의도이면 원문 기반 내용을 가진 새 판단 결과. 실질적인
        내용이 없으면 unknown fallback. 다른 의도이면 기존 판단 결과.
    """
    if decision.intent != "answer_content":
        return decision

    original_content = _extract_answer_content(response_text)
    if original_content is None:
        original_content = response_text.strip()
    if not _has_substantive_content(original_content):
        return _unknown_fallback()

    return ConfirmationIntentDecision(
        intent="answer_content",
        answer_content=original_content,
        confidence=decision.confidence,
    )


def _unknown_fallback() -> ConfirmationIntentDecision:
    """분류할 수 없는 응답에 사용할 안전한 판단을 반환한다.

    Returns:
        자동 제출이나 답변 연결을 시작하지 않는 확신도 0의 unknown 결정.
    """
    return ConfirmationIntentDecision(
        intent="unknown",
        confidence=0.0,
    )
