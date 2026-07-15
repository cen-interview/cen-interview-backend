"""WebSocket 입력과 답변 완료·확인 대화 상태를 조율한다."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass

from interview.config import settings
from interview.interviewer.facade import InterviewSession
from interview.interviewer.models import DeliveryMetrics
from interview.interviewer.session import SessionState
from interview.interviewer.turn_completion.buffer import VoiceTurnBuffer
from interview.interviewer.turn_completion.confirmation import (
    ConfirmationIntentClassifier,
)
from interview.interviewer.turn_completion.judge import TurnCompletionJudge
from interview.interviewer.turn_completion.models import (
    ConfirmationIntentDecision,
    TurnCompletionContextTurn,
    TurnCompletionDecision,
    TurnCompletionQuestionSnapshot,
    TurnCompletionResult,
    TurnCompletionSnapshot,
)
from interview.interviewer.turn_completion.registry import VoiceTurnRegistryEntry
from interview.interviewer.turn_completion.worker import (
    LatestWinsTurnCompletionWorker,
    TurnCompletionResultCallback,
)
from interview.schemas.events import Mode


CONFIRMATION_PROMPT_TEXT = "네, 답변은 여기까지일까요?"
"""애매한 답변 종료 여부를 묻는 고정 확인 문구."""

ConfirmationRequestedCallback = Callable[[str, str, int, str], Awaitable[None]]
"""confirmation ID, 질문 ID, revision과 문구를 전달하는 비동기 callback."""

ConfirmationCancelledCallback = Callable[[str, str, str], Awaitable[None]]
"""confirmation ID, 질문 ID와 취소 사유를 전달하는 비동기 callback."""

TurnStateChangedCallback = Callable[[str, int, str], Awaitable[None]]
"""질문 ID, revision과 계속 듣기 사유를 전달하는 비동기 callback."""


@dataclass(slots=True)
class ConfirmationResponseResult:
    """확인 응답 분류와 적용 이후의 음성 턴 상태를 묶는다.

    Attributes:
        buffer:
            확인 응답 의도를 적용한 직후의 VoiceTurnBuffer 복사본.

        decision:
            finish, continue, answer_content 또는 unknown 확인 응답 의도.
    """

    buffer: VoiceTurnBuffer
    decision: ConfirmationIntentDecision


class VoiceTurnCoordinatorError(ValueError):
    """음성 턴 입력을 현재 면접 상태에 적용할 수 없을 때 발생하는 오류.

    Attributes:
        code:
            WebSocket 오류 응답으로 변환할 안정적인 오류 코드.

        recoverable:
            현재 연결에서 다음 메시지를 계속 처리할 수 있는지 여부.
    """

    def __init__(self, code: str, message: str, *, recoverable: bool) -> None:
        """클라이언트용 오류 코드와 복구 가능 여부를 가진 예외를 생성한다.

        Args:
            code:
                WebSocket 오류 응답에 사용할 안정적인 코드.

            message:
                내부 구현 정보를 제외한 사용자 노출 가능 설명.

            recoverable:
                같은 연결에서 후속 이벤트를 계속 처리할 수 있는지 여부.
        """
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


class VoiceTurnCoordinator:
    """현재 음성 질문의 완료 판단과 확인 대화 상태를 조율한다.

    WebSocket transport 모델에 의존하지 않고 검증된 원시 필드만 받는다. 최신
    판단이 애매하고 확인 조건을 만족하면 짧은 대화 간격 후 고정 확인 질문을
    요청한다. 확인 준비·재생 중 새 발화가 들어오면 사용자 음성을 우선해
    confirmation을 취소한다.

    Attributes:
        _session:
            확정된 Interviewer 그래프 상태를 제공하는 현재 면접 세션.

        _entry:
            현재 음성 질문의 buffer, 비동기 lock과 worker를 보관할 registry 항목.

        _worker:
            최신 snapshot만 완료 판단하는 질문 단위 비동기 worker.

        _confirmation_task:
            확인 질문을 시작하기 전 자연스러운 대화 간격을 기다리는 task.

        _invalid_confirmation_ids:
            취소되거나 이미 처리돼 늦은 응답을 거절할 confirmation ID 집합.
    """

    def __init__(
        self,
        *,
        session: InterviewSession,
        entry: VoiceTurnRegistryEntry,
        judge: TurnCompletionJudge | None = None,
        confirmation_classifier: ConfirmationIntentClassifier | None = None,
        on_result: TurnCompletionResultCallback | None = None,
        on_state_changed: TurnStateChangedCallback | None = None,
        on_confirmation_requested: ConfirmationRequestedCallback | None = None,
        on_confirmation_cancelled: ConfirmationCancelledCallback | None = None,
        confirmation_pause_seconds: float | None = None,
        confirmation_confidence: float | None = None,
        max_confirmations: int | None = None,
    ) -> None:
        """현재 세션과 질문의 음성 턴 coordinator를 생성한다.

        Args:
            session:
                현재 면접의 확정 상태를 제공하는 InterviewSession.

            entry:
                현재 질문의 VoiceTurnBuffer와 세션별 asyncio.Lock을 가진
                registry 항목.

            judge:
                선택적으로 주입할 완료 판단기. 없으면 기본 판단기를 생성한다.

            confirmation_classifier:
                확인 응답의 finish, continue, answer_content, unknown 의도를
                분류할 판정기. 없으면 기본 판정기를 생성한다.

            on_result:
                최신 완료 판단이 buffer에 기록된 뒤 호출할 선택적 callback.
                자동 제출 연결은 다음 단계에서 이 경계를 사용할 수 있다.

            on_state_changed:
                계속 듣기 상태와 사유를 WebSocket 계층에 전달할 callback.

            on_confirmation_requested:
                고정 확인 문구의 TTS 재생을 요청할 callback.

            on_confirmation_cancelled:
                프론트가 진행 중인 확인 TTS를 중단하도록 알릴 callback.

            confirmation_pause_seconds:
                확인 질문 시작 전 기다릴 자연스러운 대화 간격.

            confirmation_confidence:
                ambiguous 판단으로 확인 질문을 시작할 최소 확신도.

            max_confirmations:
                질문 하나에서 실제 시작할 수 있는 최대 확인 질문 횟수.

        Raises:
            ValueError:
                확인 대기 시간, confidence 또는 최대 횟수가 허용 범위를 벗어난
                경우.
        """
        configured_pause = (
            settings.turn_confirmation_pause_seconds
            if confirmation_pause_seconds is None
            else confirmation_pause_seconds
        )
        configured_confidence = (
            settings.turn_completion_confirm_confidence
            if confirmation_confidence is None
            else confirmation_confidence
        )
        configured_max_confirmations = (
            settings.turn_confirmation_max_per_question
            if max_confirmations is None
            else max_confirmations
        )
        if configured_pause < 0:
            raise ValueError("확인 질문 대화 간격은 음수일 수 없습니다.")
        if not 0 <= configured_confidence <= 1:
            raise ValueError("확인 질문 confidence는 0 이상 1 이하여야 합니다.")
        if configured_max_confirmations <= 0:
            raise ValueError("질문당 확인 횟수는 0보다 커야 합니다.")

        self._session = session
        self._entry = entry
        self._confirmation_classifier = (
            confirmation_classifier or ConfirmationIntentClassifier()
        )
        self._external_on_result = on_result
        self._on_state_changed = on_state_changed
        self._on_confirmation_requested = on_confirmation_requested
        self._on_confirmation_cancelled = on_confirmation_cancelled
        self._confirmation_pause_seconds = configured_pause
        self._confirmation_confidence = configured_confidence
        self._max_confirmations = configured_max_confirmations
        self._confirmation_task: asyncio.Task[None] | None = None
        self._invalid_confirmation_ids: set[str] = set()
        self._closed = False
        self._worker = LatestWinsTurnCompletionWorker(
            judge=judge or TurnCompletionJudge(),
            buffer=entry.buffer,
            buffer_lock=entry.lock,
            on_result=self._handle_worker_result,
        )

    @property
    def worker(self) -> LatestWinsTurnCompletionWorker:
        """registry에 연결할 현재 질문의 latest-wins worker를 반환한다."""
        return self._worker

    async def prepare_connection(self) -> VoiceTurnBuffer:
        """재연결에 남은 이전 confirmation 상태를 안전하게 정리한다.

        WebSocket 단절 이후에는 확인 TTS 재생 상태를 신뢰할 수 없으므로 준비
        또는 응답 대기 상태를 listening으로 되돌린다. 실제 시작된 확인 횟수와
        현재 답변 텍스트는 유지한다.

        Returns:
            재연결 정리 이후의 VoiceTurnBuffer 복사본.
        """
        self._cancel_confirmation_task()
        async with self._entry.lock:
            confirmation_id = self._entry.buffer.cancel_confirmation()
            if confirmation_id is not None:
                self._remember_invalid_confirmation(confirmation_id)
            return self._entry.buffer.model_copy(deep=True)

    async def handle_transcript_updated(
        self,
        *,
        question_id: str,
        revision: int,
        text: str,
        speech_active: bool,
        segment_final: bool,
        answer_duration_seconds: float | None = None,
        delivery_metrics: DeliveryMetrics | None = None,
    ) -> VoiceTurnBuffer:
        """누적 전사문 최신본을 반영하고 완료 판단 후보로 제출한다.

        확인 준비 또는 재생 중 더 높은 revision의 전사문이 도착하면 confirmation을
        취소하고 사용자 답변 수집을 우선한다.

        Args:
            question_id:
                전사문이 속한 현재 질문 ID.

            revision:
                질문 안에서 단조 증가하는 누적 전사문 버전.

            text:
                delta가 아닌 현재까지 누적된 답변 최신본.

            speech_active:
                메시지 생성 시점에 사용자가 발화 중인지 여부.

            segment_final:
                현재 STT 구간이 안정화된 최종 구간인지 여부.

            answer_duration_seconds:
                답변 시작 후 경과한 선택적 시간.

            delivery_metrics:
                현재까지 수집된 선택적 음성 전달 지표.

        Returns:
            입력 반영 직후의 VoiceTurnBuffer 복사본.

        Raises:
            VoiceTurnCoordinatorError:
                worker가 닫혔거나, 면접 상태가 음성 입력을 받을 수 없거나,
                revision이 최신값보다 높지 않은 경우.
        """
        self._ensure_worker_available()
        session_state = await self._get_active_voice_state()
        self._validate_current_question(session_state, question_id)

        cancelled_confirmation_id: str | None = None
        async with self._entry.lock:
            previous_state = self._entry.buffer.state
            cancelled_confirmation_id = self._entry.buffer.active_confirmation_id
            updated = self._entry.buffer.update_transcript(
                question_id=question_id,
                revision=revision,
                text=text,
                speech_active=speech_active,
                segment_final=segment_final,
                answer_duration_seconds=answer_duration_seconds,
                delivery_metrics=delivery_metrics,
            )
            if not updated:
                raise VoiceTurnCoordinatorError(
                    "stale_revision",
                    "최신 전사문보다 오래된 이벤트입니다.",
                    recoverable=True,
                )
            snapshot = self._build_snapshot(session_state)
            buffer_snapshot = self._entry.buffer.model_copy(deep=True)

        if previous_state in {"confirmation_pending", "confirming_end"}:
            self._cancel_confirmation_task()
            if cancelled_confirmation_id is not None:
                self._remember_invalid_confirmation(cancelled_confirmation_id)
                await self._notify_confirmation_cancelled(
                    confirmation_id=cancelled_confirmation_id,
                    question_id=question_id,
                    reason="candidate_resumed_speaking",
                )

        await self._worker.submit(snapshot)
        return buffer_snapshot

    async def handle_activity_changed(
        self,
        *,
        question_id: str,
        revision: int,
        speech_active: bool,
    ) -> VoiceTurnBuffer:
        """전사문을 변경하지 않고 현재 발화 상태를 우선 반영한다.

        speech_active가 True이면 준비 또는 재생 중인 confirmation을 취소해
        프론트가 확인 TTS를 즉시 중단할 수 있게 한다.

        Args:
            question_id:
                발화 상태 이벤트가 속한 현재 질문 ID.

            revision:
                발화 상태 이벤트가 참조하는 전사문 revision.

            speech_active:
                사용자가 현재 실제로 발화 중인지 여부.

        Returns:
            발화 상태 반영 직후의 VoiceTurnBuffer 복사본.
        """
        self._ensure_worker_available()
        session_state = await self._get_active_voice_state()
        self._validate_current_question(session_state, question_id)

        cancelled_confirmation_id: str | None = None
        async with self._entry.lock:
            previous_state = self._entry.buffer.state
            cancelled_confirmation_id = self._entry.buffer.active_confirmation_id
            updated = self._entry.buffer.update_speech_activity(
                question_id=question_id,
                revision=revision,
                speech_active=speech_active,
            )
            if not updated:
                raise VoiceTurnCoordinatorError(
                    "stale_revision",
                    "현재 전사문 revision과 다른 발화 상태 이벤트입니다.",
                    recoverable=True,
                )
            buffer_snapshot = self._entry.buffer.model_copy(deep=True)

        if speech_active and previous_state in {
            "confirmation_pending",
            "confirming_end",
        }:
            self._cancel_confirmation_task()
            if cancelled_confirmation_id is not None:
                self._remember_invalid_confirmation(cancelled_confirmation_id)
                await self._notify_confirmation_cancelled(
                    confirmation_id=cancelled_confirmation_id,
                    question_id=question_id,
                    reason="candidate_resumed_speaking",
                )
        return buffer_snapshot

    async def handle_confirmation_response(
        self,
        *,
        confirmation_id: str,
        question_id: str,
        revision: int,
        response_revision: int,
        text: str,
    ) -> ConfirmationResponseResult:
        """활성 종료 확인에 대한 지원자 응답을 분류하고 반영한다.

        확인 응답 분류 중 새 발화로 confirmation이 취소될 수 있으므로 LLM 호출
        전후에 ID, 질문과 revision을 다시 검증한다. 제어 응답은 transcript에
        넣지 않고 answer_content만 기존 답변에 연결한다.

        Args:
            confirmation_id:
                응답 대상인 활성 확인 질문 ID.

            question_id:
                확인 질문이 속한 현재 면접 질문 ID.

            revision:
                확인 질문이 시작된 원래 답변 revision.

            response_revision:
                확인 응답 STT의 새 revision. 추가 답변일 때만 적용한다.

            text:
                확인 질문 이후 지원자가 말한 응답 원문.

        Returns:
            확인 응답 의도와 적용 이후의 buffer 상태.

        Raises:
            VoiceTurnCoordinatorError:
                취소됐거나 현재 활성 confirmation과 일치하지 않는 응답인 경우.
        """
        self._ensure_worker_available()
        if confirmation_id in self._invalid_confirmation_ids:
            raise self._stale_confirmation_error()

        session_state = await self._get_active_voice_state()
        self._validate_current_question(session_state, question_id)
        async with self._entry.lock:
            self._validate_active_confirmation(
                confirmation_id=confirmation_id,
                question_id=question_id,
                revision=revision,
            )

        decision = await self._confirmation_classifier.classify(text)

        session_state = await self._get_active_voice_state()
        self._validate_current_question(session_state, question_id)
        snapshot: TurnCompletionSnapshot | None = None
        async with self._entry.lock:
            self._validate_active_confirmation(
                confirmation_id=confirmation_id,
                question_id=question_id,
                revision=revision,
            )
            self._entry.buffer.apply_confirmation_intent(
                question_id=question_id,
                confirmation_id=confirmation_id,
                expected_revision=revision,
                decision=decision,
                new_revision=(
                    response_revision if decision.intent == "answer_content" else None
                ),
            )
            self._remember_invalid_confirmation(confirmation_id)
            if decision.intent == "answer_content":
                snapshot = self._build_snapshot(session_state)
            buffer_snapshot = self._entry.buffer.model_copy(deep=True)

        if snapshot is not None:
            await self._worker.submit(snapshot)
        return ConfirmationResponseResult(
            buffer=buffer_snapshot,
            decision=decision,
        )

    async def aclose(self) -> None:
        """확인 대기 task와 현재 연결의 완료 판단 worker를 종료한다."""
        self._closed = True
        task = self._confirmation_task
        self._cancel_confirmation_task()
        if task is not None and task is not asyncio.current_task():
            with suppress(asyncio.CancelledError):
                await task
        await self._worker.aclose()

    async def _handle_worker_result(self, result: TurnCompletionResult) -> None:
        """최신 완료 판단을 계속 듣기 또는 확인 준비 정책에 적용한다.

        Args:
            result:
                worker가 현재 buffer에 실제 기록한 최신 완료 판단.
        """
        schedule_confirmation = False
        listening_reason: str | None = None
        async with self._entry.lock:
            buffer = self._entry.buffer
            if (
                self._closed
                or buffer.state != "listening"
                or buffer.question_id != result.question_id
                or buffer.revision != result.revision
                or buffer.latest_decision_revision != result.revision
            ):
                return

            decision = result.decision
            if decision.recommended_action == "keep_listening":
                listening_reason = decision.reason_code
            elif decision.recommended_action == "ask_confirmation":
                if self._can_start_confirmation(buffer, decision):
                    schedule_confirmation = buffer.mark_confirmation_pending(
                        expected_revision=result.revision,
                        max_confirmations=self._max_confirmations,
                    )
                if not schedule_confirmation:
                    listening_reason = "confirmation_not_available"

        if schedule_confirmation:
            self._schedule_confirmation(
                question_id=result.question_id,
                revision=result.revision,
            )
        elif listening_reason is not None:
            await self._notify_state_changed(
                question_id=result.question_id,
                revision=result.revision,
                reason=listening_reason,
            )

        if self._external_on_result is not None:
            await self._external_on_result(result)

    def _can_start_confirmation(
        self,
        buffer: VoiceTurnBuffer,
        decision: TurnCompletionDecision,
    ) -> bool:
        """현재 최신 판단과 buffer가 확인 질문 시작 조건을 만족하는지 확인한다.

        Args:
            buffer:
                최신 판단이 기록된 현재 VoiceTurnBuffer.

            decision:
                현재 revision의 TurnCompletionDecision.

        Returns:
            답변·STT·발화·confidence와 횟수 조건을 모두 만족하면 True.
        """
        return (
            bool(buffer.answer_text.strip())
            and not buffer.speech_active
            and buffer.segment_final
            and decision.semantic_state == "ambiguous"
            and decision.recommended_action == "ask_confirmation"
            and decision.linguistically_closed
            and decision.question_satisfied
            and decision.confidence >= self._confirmation_confidence
            and decision.reason_code != "insufficient_context"
            and buffer.confirmation_count < self._max_confirmations
        )

    def _schedule_confirmation(self, *, question_id: str, revision: int) -> None:
        """최신 확인 후보의 대화 간격 대기 task를 하나만 예약한다.

        Args:
            question_id:
                확인할 현재 질문 ID.

            revision:
                확인 후보 판단이 사용한 답변 revision.
        """
        self._cancel_confirmation_task()
        self._confirmation_task = asyncio.create_task(
            self._run_confirmation_delay(
                question_id=question_id,
                revision=revision,
            )
        )

    async def _run_confirmation_delay(
        self,
        *,
        question_id: str,
        revision: int,
    ) -> None:
        """자연스러운 대화 간격 후 최신 상태를 재검증하고 확인을 시작한다.

        Args:
            question_id:
                확인할 현재 질문 ID.

            revision:
                확인 후보 판단이 사용한 답변 revision.
        """
        current_task = asyncio.current_task()
        confirmation_id: str | None = None
        try:
            await asyncio.sleep(self._confirmation_pause_seconds)
            async with self._entry.lock:
                buffer = self._entry.buffer
                if (
                    self._closed
                    or self._worker.closed
                    or buffer.state != "confirmation_pending"
                    or buffer.question_id != question_id
                    or buffer.revision != revision
                    or buffer.latest_decision is None
                    or buffer.latest_decision_revision != revision
                    or not self._can_start_confirmation(
                        buffer,
                        buffer.latest_decision,
                    )
                ):
                    return

                confirmation_id = f"confirmation_{uuid.uuid4().hex[:12]}"
                buffer.begin_confirmation(
                    confirmation_id=confirmation_id,
                    expected_revision=revision,
                    max_confirmations=self._max_confirmations,
                )

            if self._on_confirmation_requested is None:
                await self._rollback_confirmation_request(
                    confirmation_id=confirmation_id,
                )
                return
            try:
                await self._on_confirmation_requested(
                    confirmation_id,
                    question_id,
                    revision,
                    CONFIRMATION_PROMPT_TEXT,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._rollback_confirmation_request(
                    confirmation_id=confirmation_id,
                )
        finally:
            if self._confirmation_task is current_task:
                self._confirmation_task = None

    async def _rollback_confirmation_request(self, *, confirmation_id: str) -> None:
        """확인 요청 전송 실패 시 활성 confirmation을 무효화한다.

        Args:
            confirmation_id:
                전송에 실패한 활성 확인 질문 ID.
        """
        async with self._entry.lock:
            if self._entry.buffer.active_confirmation_id == confirmation_id:
                self._entry.buffer.cancel_confirmation()
                self._remember_invalid_confirmation(confirmation_id)

    def _cancel_confirmation_task(self) -> None:
        """실행 중인 확인 대기 task에 취소를 요청한다."""
        task = self._confirmation_task
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()

    def _validate_active_confirmation(
        self,
        *,
        confirmation_id: str,
        question_id: str,
        revision: int,
    ) -> None:
        """응답 대상이 현재 활성 confirmation과 같은지 확인한다.

        Args:
            confirmation_id:
                클라이언트가 응답한 확인 질문 ID.

            question_id:
                확인 질문이 속한 질문 ID.

            revision:
                확인 질문이 대상으로 삼은 답변 revision.

        Raises:
            VoiceTurnCoordinatorError:
                확인 상태, ID, 질문 또는 revision이 현재값과 다른 경우.
        """
        buffer = self._entry.buffer
        if (
            buffer.state != "confirming_end"
            or buffer.question_id != question_id
            or buffer.revision != revision
            or buffer.active_confirmation_id != confirmation_id
            or buffer.active_confirmation_revision != revision
        ):
            raise self._stale_confirmation_error()

    def _validate_current_question(
        self,
        session_state: SessionState,
        question_id: str,
    ) -> None:
        """입력 질문 ID가 현재 Interviewer 질문과 같은지 확인한다.

        Args:
            session_state:
                현재 Interviewer SessionState.

            question_id:
                클라이언트 이벤트의 질문 ID.

        Raises:
            VoiceTurnCoordinatorError:
                현재 질문과 다른 경우.
        """
        current_question = session_state.current_question
        if current_question is None or current_question.question_id != question_id:
            raise VoiceTurnCoordinatorError(
                "question_mismatch",
                "현재 질문과 다른 음성 이벤트입니다.",
                recoverable=True,
            )

    def _remember_invalid_confirmation(self, confirmation_id: str) -> None:
        """취소되거나 처리된 confirmation ID를 늦은 응답 차단 목록에 넣는다.

        Args:
            confirmation_id:
                더 이상 응답을 허용하지 않을 confirmation ID.
        """
        self._invalid_confirmation_ids.add(confirmation_id)
        while len(self._invalid_confirmation_ids) > 20:
            self._invalid_confirmation_ids.pop()

    async def _notify_state_changed(
        self,
        *,
        question_id: str,
        revision: int,
        reason: str,
    ) -> None:
        """선택적 callback으로 계속 듣기 상태를 전달한다.

        Args:
            question_id:
                상태가 변경된 질문 ID.

            revision:
                상태 변경 기준 답변 revision.

            reason:
                계속 듣기 또는 확인 미실행 사유.
        """
        if self._on_state_changed is not None:
            await self._on_state_changed(question_id, revision, reason)

    async def _notify_confirmation_cancelled(
        self,
        *,
        confirmation_id: str,
        question_id: str,
        reason: str,
    ) -> None:
        """선택적 callback으로 확인 TTS 취소를 전달한다.

        Args:
            confirmation_id:
                취소할 활성 확인 질문 ID.

            question_id:
                확인 질문이 속한 현재 질문 ID.

            reason:
                확인 질문을 취소한 안정적인 사유 코드.
        """
        if self._on_confirmation_cancelled is None:
            return
        try:
            await self._on_confirmation_cancelled(
                confirmation_id,
                question_id,
                reason,
            )
        except Exception:
            return

    def _ensure_worker_available(self) -> None:
        """현재 연결의 worker가 새 이벤트를 받을 수 있는지 확인한다.

        Raises:
            VoiceTurnCoordinatorError:
                reconnect로 새 worker가 연결돼 현재 worker가 취소된 경우.
        """
        if self._closed or self._worker.closed:
            raise VoiceTurnCoordinatorError(
                "worker_not_available",
                "새 연결이 활성화되어 현재 음성 연결을 계속 사용할 수 없습니다.",
                recoverable=False,
            )

    async def _get_active_voice_state(self) -> SessionState:
        """동기 InterviewSession 상태를 비동기 경계 밖에서 조회한다.

        Returns:
            진행 중인 음성 면접의 최신 SessionState.

        Raises:
            VoiceTurnCoordinatorError:
                세션이 종료됐거나 음성 mode가 아니거나 현재 질문이 없는 경우.
        """
        session_state = await asyncio.to_thread(self._session.get_state)
        if session_state.finished:
            raise VoiceTurnCoordinatorError(
                "session_finished",
                "이미 종료된 면접 세션입니다.",
                recoverable=False,
            )
        if session_state.mode != Mode.VOICE.value:
            raise VoiceTurnCoordinatorError(
                "voice_mode_required",
                "음성 면접 세션에서만 사용할 수 있습니다.",
                recoverable=False,
            )
        if session_state.current_question is None:
            raise VoiceTurnCoordinatorError(
                "question_unavailable",
                "현재 답변할 질문이 없습니다.",
                recoverable=False,
            )
        return session_state

    def _build_snapshot(
        self,
        session_state: SessionState,
    ) -> TurnCompletionSnapshot:
        """현재 SessionState와 buffer를 완료 판단 입력 snapshot으로 만든다.

        Args:
            session_state:
                현재 질문과 확정 transcript를 가진 Interviewer SessionState.

        Returns:
            현재 질문, 최근 최대 두 턴과 부분 전사문 최신본을 담은 snapshot.
        """
        current_question = session_state.current_question
        recent_turns = [
            TurnCompletionContextTurn(role=turn.role, text=turn.text)
            for turn in session_state.transcript[-2:]
            if turn.text.strip()
        ]
        return TurnCompletionSnapshot(
            session_id=self._entry.buffer.session_id,
            question_id=self._entry.buffer.question_id,
            revision=self._entry.buffer.revision,
            question=TurnCompletionQuestionSnapshot(
                question_id=current_question.question_id,
                text=current_question.text,
                kind=(
                    current_question.kind.value
                    if hasattr(current_question.kind, "value")
                    else str(current_question.kind)
                ),
                topic=current_question.topic,
            ),
            current_answer=self._entry.buffer.answer_text,
            recent_turns=recent_turns,
            speech_active=self._entry.buffer.speech_active,
            segment_final=self._entry.buffer.segment_final,
            answer_duration_seconds=self._entry.buffer.answer_duration_seconds,
        )

    @staticmethod
    def _stale_confirmation_error() -> VoiceTurnCoordinatorError:
        """취소됐거나 현재 상태와 다른 confirmation 응답 오류를 만든다.

        Returns:
            연결을 유지하되 해당 응답만 폐기하는 stale_confirmation 오류.
        """
        return VoiceTurnCoordinatorError(
            "stale_confirmation",
            "취소됐거나 현재 상태와 일치하지 않는 확인 응답입니다.",
            recoverable=True,
        )
