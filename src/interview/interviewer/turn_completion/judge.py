"""LLM을 사용해 현재 음성 답변의 문맥상 완료 여부를 판단한다."""

import asyncio
from typing import Any

from interview.config import settings
from interview.interviewer.turn_completion.models import (
    TurnCompletionDecision,
    TurnCompletionResult,
    TurnCompletionSnapshot,
)
from interview.interviewer.turn_completion.prompts import (
    TURN_COMPLETION_SYSTEM_PROMPT,
    build_turn_completion_user_prompt,
)
from interview.llm.client import get_llm


class TurnCompletionJudge:
    """답변 품질 평가와 독립적으로 현재 발화의 완료 여부를 판단한다.

    기본 LLM은 낮은 temperature로 생성하며, 호출 제한 시간을 넘기거나
    구조화 출력 검증에 실패하면 자동 제출 대신 계속 듣기를 권장하는 안전한
    fallback을 반환한다.

    Attributes:
        _llm:
            TurnCompletionDecision 구조화 출력을 지원하는 LLM client.

        _timeout_seconds:
            한 번의 완료 판단 LLM 호출을 기다릴 최대 시간.
    """

    def __init__(
        self,
        *,
        llm: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """완료 판단기를 생성한다.

        Args:
            llm:
                선택적으로 주입할 LLM client. 없으면 공용 LLM을 temperature
                0으로 생성해 사용한다.

            timeout_seconds:
                LLM 응답을 기다릴 최대 시간. 없으면 애플리케이션 설정값을
                사용한다.

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
            raise ValueError("완료 판단 제한 시간은 0보다 커야 합니다.")

        self._llm = llm if llm is not None else get_llm(temperature=0.0)
        self._timeout_seconds = configured_timeout

    async def judge(self, snapshot: TurnCompletionSnapshot) -> TurnCompletionResult:
        """최신 전사 snapshot의 문맥상 완료 여부를 판단한다.

        지원자 답변은 신뢰할 수 없는 데이터로 분리한 prompt에 담고,
        TurnCompletionDecision 구조화 출력으로만 결과를 받는다. timeout,
        LLM 호출 오류 또는 출력 검증 오류가 발생하면 예외로 자동 제출 흐름을
        중단시키지 않고 계속 듣기 fallback을 반환한다.

        Args:
            snapshot:
                현재 질문과 누적 전사문 최신본을 담은 판단 입력.

        Returns:
            입력 question_id와 revision에 연결된 완료 판단 결과. 판단에
            실패한 경우 안전한 keep_listening 결정이 포함된다.
        """
        messages = [
            {"role": "system", "content": TURN_COMPLETION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_turn_completion_user_prompt(snapshot),
            },
        ]

        fallback_used = False
        try:
            structured_llm = self._llm.with_structured_output(
                TurnCompletionDecision
            )
            raw_decision = await asyncio.wait_for(
                structured_llm.ainvoke(messages),
                timeout=self._timeout_seconds,
            )
            decision = (
                raw_decision
                if isinstance(raw_decision, TurnCompletionDecision)
                else TurnCompletionDecision.model_validate(raw_decision)
            )
        except Exception:
            decision = _keep_listening_fallback()
            fallback_used = True

        return TurnCompletionResult(
            question_id=snapshot.question_id,
            revision=snapshot.revision,
            decision=decision,
            fallback_used=fallback_used,
        )


def _keep_listening_fallback() -> TurnCompletionDecision:
    """완료 판단 실패 시 사용할 안전한 계속 듣기 결정을 반환한다.

    Returns:
        자동 제출이나 확인 질문을 시작하지 않고 현재 답변을 계속 수집하도록
        하는 확신도 0의 판단 결과.
    """
    return TurnCompletionDecision(
        semantic_state="ambiguous",
        linguistically_closed=False,
        question_satisfied=False,
        continuation_expected="high",
        explicit_completion=False,
        recommended_action="keep_listening",
        confidence=0.0,
        reason_code="insufficient_context",
    )
