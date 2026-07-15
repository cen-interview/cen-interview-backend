"""WebSocket 입력을 실시간 음성 buffer와 완료 판단 worker에 연결한다."""

import asyncio

from interview.interviewer.facade import InterviewSession
from interview.interviewer.models import DeliveryMetrics
from interview.interviewer.session import SessionState
from interview.interviewer.turn_completion.buffer import VoiceTurnBuffer
from interview.interviewer.turn_completion.judge import TurnCompletionJudge
from interview.interviewer.turn_completion.models import (
    TurnCompletionContextTurn,
    TurnCompletionQuestionSnapshot,
    TurnCompletionSnapshot,
)
from interview.interviewer.turn_completion.registry import VoiceTurnRegistryEntry
from interview.interviewer.turn_completion.worker import (
    LatestWinsTurnCompletionWorker,
    TurnCompletionResultCallback,
)
from interview.schemas.events import Mode


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
    """현재 음성 질문의 buffer 갱신과 완료 판단 실행을 조율한다.

    WebSocket transport 모델에 의존하지 않고 검증된 원시 필드만 받는다. 현재
    InterviewSession의 질문을 확인해 TurnCompletionSnapshot을 만들고,
    LatestWinsTurnCompletionWorker가 최신 revision만 판단하도록 전달한다.

    Attributes:
        _session:
            확정된 Interviewer 그래프 상태를 제공하는 현재 면접 세션.

        _entry:
            현재 음성 질문의 buffer, 비동기 lock과 worker를 보관할 registry 항목.

        _worker:
            최신 snapshot만 완료 판단하는 질문 단위 비동기 worker.
    """

    def __init__(
        self,
        *,
        session: InterviewSession,
        entry: VoiceTurnRegistryEntry,
        judge: TurnCompletionJudge | None = None,
        on_result: TurnCompletionResultCallback | None = None,
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

            on_result:
                최신 판단이 buffer에 기록된 뒤 호출할 선택적 비동기 callback.
                WebSocket 계층은 이를 서버 메시지 전송 경계로 사용할 수 있다.
        """
        self._session = session
        self._entry = entry
        self._worker = LatestWinsTurnCompletionWorker(
            judge=judge or TurnCompletionJudge(),
            buffer=entry.buffer,
            buffer_lock=entry.lock,
            on_result=on_result,
        )

    @property
    def worker(self) -> LatestWinsTurnCompletionWorker:
        """registry에 연결할 현재 질문의 latest-wins worker를 반환한다."""
        return self._worker

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

            VoiceTurnBufferError:
                질문 ID 또는 buffer 상태가 입력과 맞지 않는 경우.
        """
        self._ensure_worker_available()
        session_state = await self._get_active_voice_state()
        current_question = session_state.current_question
        if current_question is None or current_question.question_id != question_id:
            raise VoiceTurnCoordinatorError(
                "question_mismatch",
                "현재 질문과 다른 전사 이벤트입니다.",
                recoverable=True,
            )

        async with self._entry.lock:
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

        Args:
            question_id:
                발화 상태 이벤트가 속한 현재 질문 ID.

            revision:
                발화 상태 이벤트가 참조하는 전사문 revision.

            speech_active:
                사용자가 현재 실제로 발화 중인지 여부.

        Returns:
            발화 상태 반영 직후의 VoiceTurnBuffer 복사본.

        Raises:
            VoiceTurnCoordinatorError:
                worker가 닫혔거나, 현재 질문과 맞지 않거나, 이벤트 revision이
                현재 buffer revision과 다른 경우.
        """
        self._ensure_worker_available()
        session_state = await self._get_active_voice_state()
        current_question = session_state.current_question
        if current_question is None or current_question.question_id != question_id:
            raise VoiceTurnCoordinatorError(
                "question_mismatch",
                "현재 질문과 다른 발화 상태 이벤트입니다.",
                recoverable=True,
            )

        async with self._entry.lock:
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
            return self._entry.buffer.model_copy(deep=True)

    async def aclose(self) -> None:
        """현재 연결에 묶인 완료 판단 worker를 비동기로 종료한다."""
        await self._worker.aclose()

    def _ensure_worker_available(self) -> None:
        """현재 연결의 worker가 새 이벤트를 받을 수 있는지 확인한다.

        Raises:
            VoiceTurnCoordinatorError:
                reconnect로 새 worker가 연결돼 현재 worker가 취소된 경우.
        """
        if self._worker.closed:
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
