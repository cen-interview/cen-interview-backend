"""WebSocket 입력과 답변 완료·확인 대화 상태를 조율한다."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from interview.config import settings
from interview.interviewer.facade import InterviewSession
from interview.interviewer.models import DeliveryMetrics
from interview.interviewer.session import SessionState
from interview.interviewer.turn_completion.buffer import (
    VoiceTurnBuffer,
    VoiceTurnCompletionReason,
    VoiceTurnRevisionConflictError,
)
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
from interview.interviewer.turn_completion.telemetry import (
    elapsed_milliseconds,
    log_voice_turn_event,
    monotonic_time,
)
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


@dataclass(frozen=True, slots=True)
class VoiceTurnCommitRequest:
    """기존 음성 제출 경로에 전달할 확정 답변 snapshot.

    Attributes:
        session_id:
            답변을 제출할 면접 세션 ID.

        question_id:
            제출 대상인 현재 질문 ID.

        revision:
            제출 직전 검증을 통과한 전사문 revision.

        answer_text:
            앞뒤 공백을 제거한 최종 답변 원문.

        completion_reason:
            자동 제출 후보를 만든 문맥 기반 사유.

        delivery_metrics:
            기존 from_voice()로 전달할 최신 음성 전달 지표.

        client_event_id:
            같은 자동 제출의 그래프 중복 실행을 막는 결정적 멱등성 ID.
    """

    session_id: str
    question_id: str
    revision: int
    answer_text: str
    completion_reason: VoiceTurnCompletionReason
    delivery_metrics: DeliveryMetrics | None
    client_event_id: str


@dataclass(frozen=True, slots=True)
class VoiceTurnCommitResult:
    """기존 제출 경로 실행 결과와 WebSocket 세션 payload를 묶는다.

    Attributes:
        request:
            제출에 사용한 불변 답변 snapshot.

        session:
            기존 세션 events API와 같은 JSON 직렬화 가능 응답.
    """

    request: VoiceTurnCommitRequest
    session: dict[str, Any]


CommitAnswerCallback = Callable[
    [VoiceTurnCommitRequest],
    Awaitable[dict[str, Any]],
]
"""확정 답변을 기존 제출 경로로 보내고 세션 payload를 반환하는 callback."""

CommitStartedCallback = Callable[[VoiceTurnCommitRequest], Awaitable[None]]
"""제출이 확정돼 되돌릴 수 없는 시점을 그래프 실행 전에 전달하는 callback."""

AnswerCommittedCallback = Callable[[VoiceTurnCommitResult], Awaitable[None]]
"""buffer 확정 후 answer.committed 메시지를 전달하는 callback."""

CommitFailedCallback = Callable[[VoiceTurnCommitRequest, str], Awaitable[None]]
"""기존 제출 경로 실패를 transport 오류로 전달하는 callback."""


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


@dataclass(frozen=True, slots=True)
class _DecisionPolicyResult:
    """최신 완료 판단을 현재 buffer 상태에 적용한 결과.

    발화 종료를 기다려야 하면 모든 필드가 기본값인 결과를 반환한다.

    Attributes:
        schedule_confirmation:
            종료 확인 질문을 대화 간격 이후 시작해야 하는지 여부.

        schedule_commit:
            자동 제출 유예 task를 시작해야 하는지 여부.

        completion_reason:
            자동 제출 후보를 만든 문맥 기반 완료 사유.

        listening_reason:
            자동 제출이나 확인 질문을 시작하지 않고 계속 듣는 사유.
    """

    schedule_confirmation: bool = False
    schedule_commit: bool = False
    completion_reason: VoiceTurnCompletionReason | None = None
    listening_reason: str | None = None


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

        _commit_task:
            완료 후보 이후 경합 방지 구간과 기존 제출 경로를 실행하는 task.

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
        on_commit_started: CommitStartedCallback | None = None,
        on_commit_answer: CommitAnswerCallback | None = None,
        on_answer_committed: AnswerCommittedCallback | None = None,
        on_commit_failed: CommitFailedCallback | None = None,
        confirmation_pause_seconds: float | None = None,
        confirmation_confidence: float | None = None,
        commit_grace_milliseconds: int | None = None,
        auto_submit_confidence: float | None = None,
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

            on_commit_started:
                제출이 확정돼 취소할 수 없는 시점을 그래프 실행 전에 전달할
                callback. 리액션 발화처럼 제출 완료를 기다리지 않는 즉시
                알림에 사용한다.

            on_commit_answer:
                확정 답변을 기존 from_voice와 submit_event 경로로 전달할 callback.

            on_answer_committed:
                buffer commit 성공 이후 WebSocket 결과를 전달할 callback.

            on_commit_failed:
                기존 제출 경로 실패를 WebSocket 오류로 전달할 callback.

            confirmation_pause_seconds:
                확인 질문 시작 전 기다릴 자연스러운 대화 간격.

            confirmation_confidence:
                ambiguous 판단으로 확인 질문을 시작할 최소 확신도.

            commit_grace_milliseconds:
                완료 후보 이후 늦은 전사·발화 패킷을 기다릴 짧은 유예 시간.

            auto_submit_confidence:
                complete 판단을 자동 제출 후보로 만들 최소 확신도.

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
        configured_commit_grace = (
            settings.turn_commit_grace_milliseconds
            if commit_grace_milliseconds is None
            else commit_grace_milliseconds
        )
        configured_auto_submit_confidence = (
            settings.turn_completion_auto_submit_confidence
            if auto_submit_confidence is None
            else auto_submit_confidence
        )
        if configured_pause < 0:
            raise ValueError("확인 질문 대화 간격은 음수일 수 없습니다.")
        if not 0 <= configured_confidence <= 1:
            raise ValueError("확인 질문 confidence는 0 이상 1 이하여야 합니다.")
        if configured_max_confirmations <= 0:
            raise ValueError("질문당 확인 횟수는 0보다 커야 합니다.")
        if configured_commit_grace < 0:
            raise ValueError("자동 제출 유예 시간은 음수일 수 없습니다.")
        if not 0 <= configured_auto_submit_confidence <= 1:
            raise ValueError("자동 제출 confidence는 0 이상 1 이하여야 합니다.")

        self._session = session
        self._entry = entry
        self._confirmation_classifier = (
            confirmation_classifier or ConfirmationIntentClassifier()
        )
        self._external_on_result = on_result
        self._on_state_changed = on_state_changed
        self._on_confirmation_requested = on_confirmation_requested
        self._on_confirmation_cancelled = on_confirmation_cancelled
        self._on_commit_started = on_commit_started
        self._on_commit_answer = on_commit_answer
        self._on_answer_committed = on_answer_committed
        self._on_commit_failed = on_commit_failed
        self._confirmation_pause_seconds = configured_pause
        self._confirmation_confidence = configured_confidence
        self._max_confirmations = configured_max_confirmations
        self._commit_grace_seconds = configured_commit_grace / 1000
        self._auto_submit_confidence = configured_auto_submit_confidence
        self._confirmation_task: asyncio.Task[None] | None = None
        self._commit_task: asyncio.Task[None] | None = None
        self._invalid_confirmation_ids: set[str] = set()
        self._accept_equal_reconnect_snapshot = False
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
        """재연결에 남은 이전 임시 판단·확인 상태를 안전하게 정리한다.

        WebSocket 단절 이후에는 확인 TTS 재생 상태를 신뢰할 수 없으므로 준비
        또는 응답 대기 상태와 미실행 완료 후보를 listening으로 되돌린다. 실제
        시작된 확인 횟수, 현재 답변 텍스트와 revision은 유지하고 같은 revision의
        보존 snapshot을 연결 직후 한 번 다시 받을 수 있게 한다.

        Returns:
            재연결 정리 이후의 VoiceTurnBuffer 복사본.
        """
        self._cancel_confirmation_task()
        self._cancel_commit_task()
        async with self._entry.lock:
            buffer = self._entry.buffer
            confirmation_id = buffer.cancel_confirmation()
            if confirmation_id is not None:
                self._remember_invalid_confirmation(confirmation_id)
            if buffer.state == "complete_candidate":
                buffer.resume_listening()
            elif buffer.state == "listening" and (
                buffer.latest_decision is not None
                or buffer.pending_completion_reason is not None
            ):
                buffer.resume_listening()
            self._accept_equal_reconnect_snapshot = (
                buffer.state == "listening"
                and (buffer.revision > 0 or bool(buffer.answer_text))
            )
            buffer_snapshot = buffer.model_copy(deep=True)

        log_voice_turn_event(
            "voice_turn.connection.prepared",
            session_id=buffer_snapshot.session_id,
            question_id=buffer_snapshot.question_id,
            revision=buffer_snapshot.revision,
            answer_text=buffer_snapshot.answer_text,
            voice_turn_state=buffer_snapshot.state,
            reconnect_snapshot_expected=self._accept_equal_reconnect_snapshot,
            confirmation_count=buffer_snapshot.confirmation_count,
        )
        return buffer_snapshot

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
        취소하고 사용자 답변 수집을 우선한다. 자동 제출 유예 중 내용이 같은 최종
        전사가 더 높은 revision으로 다시 오면 기존 판단을 새 revision으로 승계해
        중복 STT 이벤트가 제출 후보를 영구적으로 취소하지 않게 한다.

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
        policy_result: _DecisionPolicyResult | None = None
        policy_decision: TurnCompletionDecision | None = None
        equivalent_completion_candidate = False
        async with self._entry.lock:
            buffer = self._entry.buffer
            previous_state = buffer.state
            previous_revision = buffer.revision
            previous_answer_text = buffer.answer_text
            previous_decision_revision = buffer.latest_decision_revision
            previous_decision = buffer.latest_decision
            cancelled_confirmation_id = buffer.active_confirmation_id
            reconnect_snapshot = (
                self._accept_equal_reconnect_snapshot
                and revision == buffer.revision
            )
            if reconnect_snapshot:
                try:
                    buffer.synchronize_reconnect_snapshot(
                        question_id=question_id,
                        revision=revision,
                        text=text,
                        speech_active=speech_active,
                        segment_final=segment_final,
                        answer_duration_seconds=answer_duration_seconds,
                        delivery_metrics=delivery_metrics,
                    )
                except VoiceTurnRevisionConflictError as exc:
                    log_voice_turn_event(
                        "voice_turn.connection.revision_conflict",
                        session_id=buffer.session_id,
                        question_id=question_id,
                        revision=revision,
                        answer_text=text,
                        server_revision=buffer.revision,
                    )
                    raise VoiceTurnCoordinatorError(
                        "revision_conflict",
                        str(exc),
                        recoverable=True,
                    ) from exc
                updated = True
                self._accept_equal_reconnect_snapshot = False
            else:
                updated = buffer.update_transcript(
                    question_id=question_id,
                    revision=revision,
                    text=text,
                    speech_active=speech_active,
                    segment_final=segment_final,
                    answer_duration_seconds=answer_duration_seconds,
                    delivery_metrics=delivery_metrics,
                )
                if updated:
                    self._accept_equal_reconnect_snapshot = False
            if not updated:
                log_voice_turn_event(
                    "voice_turn.transcript.discarded",
                    session_id=buffer.session_id,
                    question_id=question_id,
                    revision=revision,
                    answer_text=text,
                    discard_reason="stale_revision",
                    server_revision=buffer.revision,
                )
                raise VoiceTurnCoordinatorError(
                    "stale_revision",
                    "최신 전사문보다 오래된 이벤트입니다.",
                    recoverable=True,
                )
            equivalent_completion_candidate = (
                not reconnect_snapshot
                and previous_state == "complete_candidate"
                and previous_decision is not None
                and previous_decision_revision == previous_revision
                and " ".join(previous_answer_text.split())
                == " ".join(text.split())
                and not speech_active
                and segment_final
                and previous_decision.recommended_action == "auto_submit"
            )
            if equivalent_completion_candidate:
                carried_result = TurnCompletionResult(
                    question_id=question_id,
                    revision=revision,
                    decision=previous_decision,
                    fallback_used=False,
                )
                buffer.record_decision(carried_result)
                policy_decision = previous_decision
                policy_result = self._apply_decision_policy(
                    buffer=buffer,
                    decision=policy_decision,
                    revision=revision,
                )
            snapshot = self._build_snapshot(session_state)
            buffer_snapshot = buffer.model_copy(deep=True)

        if reconnect_snapshot:
            log_voice_turn_event(
                "voice_turn.connection.snapshot_resynchronized",
                session_id=buffer_snapshot.session_id,
                question_id=buffer_snapshot.question_id,
                revision=buffer_snapshot.revision,
                answer_text=buffer_snapshot.answer_text,
                speech_active=buffer_snapshot.speech_active,
                segment_final=buffer_snapshot.segment_final,
            )

        if previous_state in {"confirmation_pending", "confirming_end"}:
            self._cancel_confirmation_task()
            if cancelled_confirmation_id is not None:
                self._remember_invalid_confirmation(cancelled_confirmation_id)
                await self._notify_confirmation_cancelled(
                    confirmation_id=cancelled_confirmation_id,
                    question_id=question_id,
                    reason="candidate_resumed_speaking",
                )
        if previous_state == "complete_candidate":
            self._cancel_commit_task()
            log_voice_turn_event(
                "voice_turn.commit.cancelled",
                session_id=buffer_snapshot.session_id,
                question_id=buffer_snapshot.question_id,
                revision=buffer_snapshot.revision,
                answer_text=buffer_snapshot.answer_text,
                cancel_reason=(
                    "equivalent_transcript_revision"
                    if equivalent_completion_candidate
                    else "new_transcript"
                ),
                candidate_resumed_speaking=speech_active,
            )

        if policy_result is not None and policy_decision is not None:
            await self._dispatch_decision_policy(
                policy_result=policy_result,
                buffer_snapshot=buffer_snapshot,
                decision=policy_decision,
                decision_trigger="equivalent_transcript",
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
        프론트가 확인 TTS를 즉시 중단할 수 있게 한다. True에서 False로 바뀌면
        같은 revision에 기록된 최신 완료 판단을 현재 발화 상태에 다시 적용해
        LLM 판단과 발화 종료 이벤트의 도착 순서에 따른 자동 제출 누락을 막는다.

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
        policy_result: _DecisionPolicyResult | None = None
        policy_decision: TurnCompletionDecision | None = None
        async with self._entry.lock:
            buffer = self._entry.buffer
            previous_state = buffer.state
            previous_speech_active = buffer.speech_active
            cancelled_confirmation_id = buffer.active_confirmation_id
            updated = buffer.update_speech_activity(
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
            if (
                previous_speech_active
                and not speech_active
                and buffer.state == "listening"
                and buffer.segment_final
                and buffer.latest_decision_revision == revision
                and buffer.latest_decision is not None
                and (
                    buffer.latest_decision.recommended_action
                    in {"auto_submit", "ask_confirmation"}
                )
            ):
                policy_decision = buffer.latest_decision
                policy_result = self._apply_decision_policy(
                    buffer=buffer,
                    decision=policy_decision,
                    revision=revision,
                )
            buffer_snapshot = buffer.model_copy(deep=True)

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
        if speech_active and previous_state == "complete_candidate":
            self._cancel_commit_task()
            log_voice_turn_event(
                "voice_turn.commit.cancelled",
                session_id=buffer_snapshot.session_id,
                question_id=buffer_snapshot.question_id,
                revision=buffer_snapshot.revision,
                answer_text=buffer_snapshot.answer_text,
                cancel_reason="candidate_resumed_speaking",
                candidate_resumed_speaking=True,
            )
        if policy_result is not None and policy_decision is not None:
            await self._dispatch_decision_policy(
                policy_result=policy_result,
                buffer_snapshot=buffer_snapshot,
                decision=policy_decision,
                decision_trigger="speech_inactive",
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
        elif decision.intent == "finish":
            log_voice_turn_event(
                "voice_turn.commit.candidate",
                session_id=buffer_snapshot.session_id,
                question_id=buffer_snapshot.question_id,
                revision=buffer_snapshot.revision,
                answer_text=buffer_snapshot.answer_text,
                completion_reason="user_confirmed",
                grace_milliseconds=round(self._commit_grace_seconds * 1000),
            )
            self._schedule_commit(
                question_id=buffer_snapshot.question_id,
                revision=buffer_snapshot.revision,
            )
        log_voice_turn_event(
            "voice_turn.confirmation.responded",
            session_id=buffer_snapshot.session_id,
            question_id=buffer_snapshot.question_id,
            revision=buffer_snapshot.revision,
            answer_text=buffer_snapshot.answer_text,
            confirmation_id=confirmation_id,
            confirmation_intent=decision.intent,
            confidence=decision.confidence,
            voice_turn_state=buffer_snapshot.state,
        )
        return ConfirmationResponseResult(
            buffer=buffer_snapshot,
            decision=decision,
        )

    async def aclose(self) -> None:
        """확인·제출 대기 task와 현재 연결의 완료 판단 worker를 종료한다."""
        self._closed = True
        confirmation_task = self._confirmation_task
        commit_task = self._commit_task
        self._cancel_confirmation_task()
        self._cancel_commit_task()
        if (
            confirmation_task is not None
            and confirmation_task is not asyncio.current_task()
        ):
            with suppress(asyncio.CancelledError):
                await confirmation_task
        if commit_task is not None and commit_task is not asyncio.current_task():
            with suppress(asyncio.CancelledError):
                await commit_task
        await self._worker.aclose()

    async def _handle_worker_result(self, result: TurnCompletionResult) -> None:
        """최신 완료 판단을 계속 듣기 또는 확인 준비 정책에 적용한다.

        Args:
            result:
                worker가 현재 buffer에 실제 기록한 최신 완료 판단.
        """
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

            policy_result = self._apply_decision_policy(
                buffer=buffer,
                decision=result.decision,
                revision=result.revision,
                fallback_used=result.fallback_used,
            )
            buffer_snapshot = buffer.model_copy(deep=True)

        await self._dispatch_decision_policy(
            policy_result=policy_result,
            buffer_snapshot=buffer_snapshot,
            decision=result.decision,
            decision_trigger="judge_result",
        )

        if self._external_on_result is not None:
            await self._external_on_result(result)

    def _apply_decision_policy(
        self,
        *,
        buffer: VoiceTurnBuffer,
        decision: TurnCompletionDecision,
        revision: int,
        fallback_used: bool = False,
    ) -> _DecisionPolicyResult:
        """최신 완료 판단을 현재 발화 상태에 맞는 상태 전이로 적용한다.

        호출자는 세션별 lock을 보유하고 buffer의 질문과 revision이 최신인지 먼저
        확인해야 한다. LLM 판단 직후와 발화 종료 activity 이벤트가 같은 정책을
        사용하게 해 두 이벤트의 도착 순서가 달라도 동일한 결과를 만든다.

        Args:
            buffer:
                최신 판단과 현재 발화 상태를 보관한 음성 턴 buffer.

            decision:
                현재 revision에 기록된 완료 판단.

            revision:
                판단과 상태 전이를 연결할 최신 전사문 revision.

            fallback_used:
                LLM 실패로 안전한 계속 듣기 판단이 사용됐는지 여부.

        Returns:
            lock 밖에서 실행할 task 예약 또는 계속 듣기 알림 정보를 담은 결과.
        """
        if decision.recommended_action == "keep_listening":
            return _DecisionPolicyResult(
                listening_reason=(
                    "completion_judge_failed"
                    if fallback_used
                    else decision.reason_code
                )
            )

        if buffer.speech_active:
            return _DecisionPolicyResult()

        if decision.recommended_action == "auto_submit":
            if not self._can_start_auto_submit(buffer, decision):
                return _DecisionPolicyResult(
                    listening_reason="auto_submit_not_available"
                )
            completion_reason: VoiceTurnCompletionReason = (
                "explicit_finish"
                if decision.explicit_completion
                else "semantic_complete"
            )
            buffer.mark_complete_candidate(
                expected_revision=revision,
                completion_reason=completion_reason,
            )
            return _DecisionPolicyResult(
                schedule_commit=True,
                completion_reason=completion_reason,
            )

        if self._can_start_confirmation(buffer, decision):
            schedule_confirmation = buffer.mark_confirmation_pending(
                expected_revision=revision,
                max_confirmations=self._max_confirmations,
            )
            if schedule_confirmation:
                return _DecisionPolicyResult(schedule_confirmation=True)
        return _DecisionPolicyResult(
            listening_reason="confirmation_not_available"
        )

    async def _dispatch_decision_policy(
        self,
        *,
        policy_result: _DecisionPolicyResult,
        buffer_snapshot: VoiceTurnBuffer,
        decision: TurnCompletionDecision,
        decision_trigger: str,
    ) -> None:
        """판단 정책의 task 예약과 WebSocket 알림을 lock 밖에서 실행한다.

        Args:
            policy_result:
                lock 안에서 현재 buffer에 적용한 판단 정책 결과.

            buffer_snapshot:
                정책 적용 직후 복사한 질문과 revision별 buffer 상태.

            decision:
                로그에 의미 상태와 confidence를 기록할 최신 완료 판단.

            decision_trigger:
                판단을 처음 적용했는지 발화 종료 후 재적용했는지 나타내는 값.
        """
        if policy_result.schedule_commit:
            log_voice_turn_event(
                "voice_turn.commit.candidate",
                session_id=buffer_snapshot.session_id,
                question_id=buffer_snapshot.question_id,
                revision=buffer_snapshot.revision,
                answer_text=buffer_snapshot.answer_text,
                completion_reason=policy_result.completion_reason,
                semantic_state=decision.semantic_state,
                confidence=decision.confidence,
                grace_milliseconds=round(self._commit_grace_seconds * 1000),
                decision_trigger=decision_trigger,
            )
            self._schedule_commit(
                question_id=buffer_snapshot.question_id,
                revision=buffer_snapshot.revision,
            )
        elif policy_result.schedule_confirmation:
            log_voice_turn_event(
                "voice_turn.confirmation.pending",
                session_id=buffer_snapshot.session_id,
                question_id=buffer_snapshot.question_id,
                revision=buffer_snapshot.revision,
                answer_text=buffer_snapshot.answer_text,
                confidence=decision.confidence,
                pause_milliseconds=round(
                    self._confirmation_pause_seconds * 1000
                ),
                decision_trigger=decision_trigger,
            )
            self._schedule_confirmation(
                question_id=buffer_snapshot.question_id,
                revision=buffer_snapshot.revision,
            )
        elif policy_result.listening_reason is not None:
            await self._notify_state_changed(
                question_id=buffer_snapshot.question_id,
                revision=buffer_snapshot.revision,
                reason=policy_result.listening_reason,
            )

    def _can_start_auto_submit(
        self,
        buffer: VoiceTurnBuffer,
        decision: TurnCompletionDecision,
    ) -> bool:
        """현재 최신 완료 판단을 자동 제출 후보로 만들 수 있는지 확인한다.

        Args:
            buffer:
                최신 판단이 기록된 현재 VoiceTurnBuffer.

            decision:
                현재 revision의 TurnCompletionDecision.

        Returns:
            답변, 발화, STT 안정화, LLM의 자동 제출 권장과 confidence 조건을
            모두 만족하면 True.
        """
        return (
            bool(buffer.answer_text.strip())
            and not buffer.speech_active
            and buffer.segment_final
            and decision.semantic_state == "complete"
            and decision.recommended_action == "auto_submit"
            and decision.confidence >= self._auto_submit_confidence
            and decision.reason_code != "insufficient_context"
        )

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

    def _schedule_commit(self, *, question_id: str, revision: int) -> None:
        """최신 완료 후보의 제출 유예 task를 하나만 예약한다.

        Args:
            question_id:
                제출 후보가 속한 현재 질문 ID.

            revision:
                제출 후보를 만든 최신 답변 revision.
        """
        self._cancel_commit_task()
        self._commit_task = asyncio.create_task(
            self._run_commit_delay(
                question_id=question_id,
                revision=revision,
            )
        )

    async def _run_commit_delay(
        self,
        *,
        question_id: str,
        revision: int,
    ) -> None:
        """짧은 경합 방지 구간 후 기존 답변 제출 경로를 실행한다.

        Args:
            question_id:
                제출 후보가 속한 현재 질문 ID.

            revision:
                제출 후보를 만든 최신 답변 revision.
        """
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(self._commit_grace_seconds)
            try:
                session_state = await self._get_active_voice_state()
                self._validate_current_question(session_state, question_id)
            except VoiceTurnCoordinatorError:
                await self._rollback_complete_candidate(
                    question_id=question_id,
                    revision=revision,
                )
                return

            async with self._entry.lock:
                buffer = self._entry.buffer
                current_question = session_state.current_question
                if (
                    self._closed
                    or self._worker.closed
                    or buffer.state != "complete_candidate"
                    or buffer.question_id != question_id
                    or buffer.revision != revision
                    or buffer.speech_active
                    or not buffer.answer_text.strip()
                    or buffer.pending_completion_reason is None
                    or current_question is None
                    or current_question.question_id != question_id
                ):
                    return

                completion_reason = buffer.pending_completion_reason
                answer_text = buffer.begin_commit(
                    question_id=question_id,
                    expected_revision=revision,
                )
                request = VoiceTurnCommitRequest(
                    session_id=buffer.session_id,
                    question_id=question_id,
                    revision=revision,
                    answer_text=answer_text,
                    completion_reason=completion_reason,
                    delivery_metrics=(
                        buffer.latest_delivery_metrics.model_copy(deep=True)
                        if buffer.latest_delivery_metrics is not None
                        else None
                    ),
                    client_event_id=(
                        f"voice:{buffer.session_id}:{question_id}:"
                        f"{revision}:{completion_reason}"
                    ),
                )

            commit_started_at = monotonic_time()
            log_voice_turn_event(
                "voice_turn.commit.started",
                session_id=request.session_id,
                question_id=request.question_id,
                revision=request.revision,
                answer_text=request.answer_text,
                completion_reason=request.completion_reason,
                client_event_id=request.client_event_id,
            )
            if self._on_commit_answer is None:
                await self._abort_commit(request=request)
                log_voice_turn_event(
                    "voice_turn.commit.failed",
                    session_id=request.session_id,
                    question_id=request.question_id,
                    revision=request.revision,
                    answer_text=request.answer_text,
                    completion_reason=request.completion_reason,
                    latency_ms=elapsed_milliseconds(commit_started_at),
                    error_code="commit_callback_unavailable",
                )
                await self._notify_state_changed(
                    question_id=question_id,
                    revision=revision,
                    reason="auto_submit_not_available",
                )
                return

            await self._notify_commit_started(request=request)

            try:
                session_payload = await self._on_commit_answer(request)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._abort_commit(request=request)
                log_voice_turn_event(
                    "voice_turn.commit.failed",
                    session_id=request.session_id,
                    question_id=request.question_id,
                    revision=request.revision,
                    answer_text=request.answer_text,
                    completion_reason=request.completion_reason,
                    latency_ms=elapsed_milliseconds(commit_started_at),
                    error_code=type(exc).__name__,
                )
                await self._notify_commit_failed(
                    request=request,
                    reason="commit_failed",
                )
                return

            async with self._entry.lock:
                self._entry.buffer.mark_committed(
                    question_id=question_id,
                    expected_revision=revision,
                )

            log_voice_turn_event(
                "voice_turn.commit.completed",
                session_id=request.session_id,
                question_id=request.question_id,
                revision=request.revision,
                answer_text=request.answer_text,
                completion_reason=request.completion_reason,
                latency_ms=elapsed_milliseconds(commit_started_at),
                client_event_id=request.client_event_id,
                session_finished=bool(session_payload.get("finished")),
            )
            self._worker.cancel()
            if self._on_answer_committed is not None:
                try:
                    await self._on_answer_committed(
                        VoiceTurnCommitResult(
                            request=request,
                            session=session_payload,
                        )
                    )
                except Exception:
                    return
        finally:
            if self._commit_task is current_task:
                self._commit_task = None

    async def _rollback_complete_candidate(
        self,
        *,
        question_id: str,
        revision: int,
    ) -> None:
        """제출 전 세션 검증 실패 시 완료 후보를 listening으로 되돌린다.

        Args:
            question_id:
                되돌릴 완료 후보의 질문 ID.

            revision:
                되돌릴 완료 후보의 답변 revision.
        """
        async with self._entry.lock:
            buffer = self._entry.buffer
            if (
                buffer.state == "complete_candidate"
                and buffer.question_id == question_id
                and buffer.revision == revision
            ):
                buffer.resume_listening()
                log_voice_turn_event(
                    "voice_turn.commit.cancelled",
                    session_id=buffer.session_id,
                    question_id=question_id,
                    revision=revision,
                    answer_text=buffer.answer_text,
                    cancel_reason="session_state_changed",
                )

    async def _abort_commit(self, *, request: VoiceTurnCommitRequest) -> None:
        """기존 제출 경로 실패 후 committing 상태를 listening으로 복구한다.

        Args:
            request:
                실패한 제출에 사용한 불변 답변 snapshot.
        """
        async with self._entry.lock:
            buffer = self._entry.buffer
            if (
                buffer.state == "committing"
                and buffer.question_id == request.question_id
                and buffer.revision == request.revision
            ):
                buffer.abort_commit(
                    question_id=request.question_id,
                    expected_revision=request.revision,
                )

    async def _notify_commit_started(
        self,
        *,
        request: VoiceTurnCommitRequest,
    ) -> None:
        """선택적 callback으로 제출 확정 시점을 그래프 실행 전에 전달한다.

        리액션 발화 같은 즉시 알림이 실패해도 실제 제출은 계속돼야 하므로
        CancelledError를 제외한 예외는 삼킨다.

        Args:
            request:
                제출이 확정된 불변 답변 snapshot.
        """
        if self._on_commit_started is None:
            return
        try:
            await self._on_commit_started(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _notify_commit_failed(
        self,
        *,
        request: VoiceTurnCommitRequest,
        reason: str,
    ) -> None:
        """선택적 callback으로 기존 제출 경로 실패를 전달한다.

        Args:
            request:
                실패한 제출에 사용한 불변 답변 snapshot.

            reason:
                클라이언트 오류로 변환할 안정적인 실패 사유.
        """
        if self._on_commit_failed is None:
            return
        try:
            await self._on_commit_failed(request, reason)
        except Exception:
            return

    def _cancel_commit_task(self) -> None:
        """아직 facade 호출 전인 자동 제출 유예 task를 취소한다.

        committing 상태에서 worker thread 호출을 취소하면 실제 그래프 제출은
        계속되면서 coroutine만 중단될 수 있으므로, 제출 진입 이후에는 취소하지
        않는다.
        """
        task = self._commit_task
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
            and self._entry.buffer.state != "committing"
        ):
            task.cancel()

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

            log_voice_turn_event(
                "voice_turn.confirmation.requested",
                session_id=self._entry.buffer.session_id,
                question_id=question_id,
                revision=revision,
                answer_text=self._entry.buffer.answer_text,
                confirmation_id=confirmation_id,
                confirmation_count=self._entry.buffer.confirmation_count,
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
        log_voice_turn_event(
            "voice_turn.confirmation.cancelled",
            session_id=self._entry.buffer.session_id,
            question_id=question_id,
            revision=self._entry.buffer.revision,
            answer_text=self._entry.buffer.answer_text,
            confirmation_id=confirmation_id,
            cancel_reason=reason,
        )
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
