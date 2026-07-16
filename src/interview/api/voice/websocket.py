"""실시간 음성 전사문과 턴 상태를 연결하는 WebSocket 라우터."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from interview.api.auth.dependency import authenticate_access_token
from interview.api.database import get_db
from interview.api.interviews.service import get_interview_session_by_runtime_id
from interview.api.sessions.router import _save_finished_result, _session_response
from interview.api.users.model import User
from interview.api.voice.schema import (
    VOICE_TURN_CLIENT_MESSAGE_ADAPTER,
    AnswerCommittedMessage,
    AnswerReactionMessage,
    AnswerTranscriptUpdatedMessage,
    ConnectionAuthenticateMessage,
    ConnectionReadyMessage,
    TurnConfirmationCancelledMessage,
    TurnConfirmationRequestedMessage,
    TurnConfirmationResponseActivityChangedMessage,
    TurnConfirmationResponseReadyMessage,
    TurnConfirmationRespondedMessage,
    TurnStateChangedMessage,
    VoiceActivityChangedMessage,
    VoiceTurnErrorMessage,
)
from interview.config import settings
from interview.interviewer.adapters import from_voice
from interview.interviewer.facade import get_session
from interview.interviewer.models import DeliveryMetrics
from interview.interviewer.speech.utterance import (
    commit_acknowledgment,
    listening_cutoff_notice,
)
from interview.interviewer.turn_completion.buffer import (
    VoiceTurnAlreadyCommittedError,
    VoiceTurnBufferError,
    VoiceTurnQuestionMismatchError,
)
from interview.interviewer.turn_completion.coordinator import (
    VoiceTurnCommitRequest,
    VoiceTurnCommitResult,
    VoiceTurnCoordinator,
    VoiceTurnCoordinatorError,
)
from interview.interviewer.turn_completion.registry import (
    VoiceTurnRegistryEntry,
    get_voice_turn_registry,
)
from interview.interviewer.turn_completion.telemetry import log_voice_turn_event
from interview.schemas.events import Mode


router = APIRouter()
logger = logging.getLogger(__name__)

_POLICY_VIOLATION_CLOSE_CODE = 1008

WebSocketModelSender = Callable[[BaseModel], Awaitable[None]]
"""Pydantic 서버 메시지를 WebSocket JSON으로 전송하는 비동기 함수 타입."""


@router.websocket("/sessions/{session_id}/voice-turn")
async def voice_turn_websocket(
    websocket: WebSocket,
    session_id: str,
    db: Session = Depends(get_db),
) -> None:
    """실시간 전사문을 현재 음성 질문의 완료 판단 pipeline에 전달한다.

    연결 직후 첫 JSON 메시지로 Access Token을 검증한다. 인증과 세션 소유권,
    음성 mode 및 현재 질문을 확인한 뒤 누적 전사문과 발화 상태 메시지를
    VoiceTurnCoordinator에 전달한다. 연결이 끊기면 worker만 정리하고 buffer는
    reconnect를 위해 유지한다.

    Args:
        websocket:
            브라우저와 양방향 JSON 메시지를 주고받을 FastAPI WebSocket.

        session_id:
            실시간 음성 답변을 진행할 면접 세션 ID.

        db:
            인증 사용자와 면접 세션 소유권을 조회할 SQLAlchemy 세션.
    """
    await websocket.accept()
    send_lock = asyncio.Lock()
    connection_open = True
    coordinator: VoiceTurnCoordinator | None = None
    reaction_sent_event_id: str | None = None
    registry = get_voice_turn_registry()

    async def send_model(message: BaseModel) -> None:
        """동시 coroutine의 WebSocket JSON 전송을 하나씩 처리한다.

        Args:
            message:
                JSON 직렬화할 서버 Pydantic 메시지.
        """
        async with send_lock:
            await websocket.send_json(message.model_dump(mode="json"))

    try:
        current_user = await _authenticate_first_message(
            websocket=websocket,
            db=db,
            send_model=send_model,
        )
        if current_user is None:
            return

        try:
            session = get_session(session_id)
        except KeyError:
            await _send_fatal_error(
                websocket=websocket,
                send_model=send_model,
                code="session_not_found",
                message="면접 세션을 찾을 수 없습니다.",
            )
            return

        db_session = get_interview_session_by_runtime_id(
            db,
            runtime_session_id=session_id,
            user_id=current_user.id,
        )
        if db_session is None:
            await _send_fatal_error(
                websocket=websocket,
                send_model=send_model,
                code="session_forbidden",
                message="이 면접 세션에 접근할 수 없습니다.",
            )
            return

        state = await asyncio.to_thread(session.get_state)
        if state.finished:
            await _send_fatal_error(
                websocket=websocket,
                send_model=send_model,
                code="session_finished",
                message="이미 종료된 면접 세션입니다.",
            )
            return
        if state.mode != Mode.VOICE.value:
            await _send_fatal_error(
                websocket=websocket,
                send_model=send_model,
                code="voice_mode_required",
                message="음성 면접 세션에서만 사용할 수 있습니다.",
            )
            return
        if state.current_question is None:
            await _send_fatal_error(
                websocket=websocket,
                send_model=send_model,
                code="question_unavailable",
                message="현재 답변할 질문이 없습니다.",
            )
            return

        entry = registry.open_turn(
            session_id=session_id,
            question_id=state.current_question.question_id,
        )
        reconnecting = entry.buffer.revision > 0 or bool(entry.buffer.answer_text)

        def owns_current_worker() -> bool:
            """현재 WebSocket이 registry의 활성 worker를 소유하는지 확인한다.

            Returns:
                연결이 열려 있고 현재 coordinator의 worker가 registry에 연결돼
                있으면 True.
            """
            if not connection_open or coordinator is None:
                return False
            try:
                current_entry = registry.get(session_id)
            except KeyError:
                return False
            return current_entry.worker is coordinator.worker

        async def on_state_changed(
            question_id: str,
            revision: int,
            reason: str,
        ) -> None:
            """최신 계속 듣기 상태를 현재 WebSocket에 전달한다.

            Args:
                question_id:
                    상태가 변경된 현재 질문 ID.

                revision:
                    상태 판단에 사용한 최신 답변 revision.

                reason:
                    계속 듣기 상태를 유지하거나 되돌린 사유.
            """
            if not owns_current_worker():
                return
            try:
                await send_model(
                    TurnStateChangedMessage(
                        question_id=question_id,
                        revision=revision,
                        reason=reason,
                    )
                )
            except Exception:
                return

        async def on_confirmation_requested(
            confirmation_id: str,
            question_id: str,
            revision: int,
            text: str,
        ) -> None:
            """확인 질문 재생 요청을 현재 WebSocket에 전달한다.

            Args:
                confirmation_id:
                    확인 요청과 후속 응답을 연결할 고유 ID.

                question_id:
                    종료 여부를 확인할 현재 질문 ID.

                revision:
                    확인 판단에 사용한 답변 revision.

                text:
                    프론트가 TTS로 재생할 고정 확인 문구.
            """
            if not owns_current_worker():
                return
            await send_model(
                TurnConfirmationRequestedMessage(
                    confirmation_id=confirmation_id,
                    question_id=question_id,
                    revision=revision,
                    text=text,
                    ready_timeout_milliseconds=round(
                        settings.turn_confirmation_ready_timeout_seconds * 1000
                    ),
                    response_timeout_milliseconds=round(
                        settings.turn_confirmation_response_timeout_seconds * 1000
                    ),
                    requires_ready_ack=True,
                )
            )

        async def on_confirmation_cancelled(
            confirmation_id: str,
            question_id: str,
            reason: str,
        ) -> None:
            """중간 발화로 취소된 확인 질문을 현재 WebSocket에 알린다.

            Args:
                confirmation_id:
                    취소할 활성 확인 질문 ID.

                question_id:
                    확인 질문이 속한 현재 질문 ID.

                reason:
                    확인 질문을 취소한 안정적인 사유.
            """
            if not owns_current_worker():
                return
            await send_model(
                TurnConfirmationCancelledMessage(
                    confirmation_id=confirmation_id,
                    question_id=question_id,
                    reason=reason,
                )
            )

        async def on_commit_started(
            request: VoiceTurnCommitRequest,
        ) -> None:
            """제출 확정 직후 중립 리액션 발화를 프론트에 먼저 전달한다.

            정체로 답변 듣기를 중단한 listening_cutoff 제출에서는 중립 리액션
            대신 답변 수집 종료를 알리는 고정 안내 문구를 전송한다.

            다음 질문 생성이 완료되기 전에 프론트가 리액션 TTS 재생을
            시작할 수 있게 answer.committed보다 앞서 전송한다. 전송에
            성공하면 해당 제출 건을 기억해 두었다가 세션 응답의
            utterance_queue에서 리액션 문장을 제거한다.

            Args:
                request:
                    제출이 확정된 불변 답변 snapshot.
            """
            nonlocal reaction_sent_event_id
            if not owns_current_worker():
                return
            reaction_text = (
                listening_cutoff_notice()
                if request.completion_reason == "listening_cutoff"
                else commit_acknowledgment()
            )
            await send_model(
                AnswerReactionMessage(
                    question_id=request.question_id,
                    revision=request.revision,
                    text=reaction_text,
                )
            )
            reaction_sent_event_id = request.client_event_id
            log_voice_turn_event(
                "voice_turn.reaction.sent",
                session_id=session_id,
                question_id=request.question_id,
                revision=request.revision,
                reaction_length=len(reaction_text),
            )

        async def on_commit_answer(
            request: VoiceTurnCommitRequest,
        ) -> dict:
            """확정 답변을 기존 음성 제출 경로로 전달한다.

            동기 InterviewSession facade와 그래프 실행은 worker thread에서
            처리해 WebSocket event loop를 막지 않는다. 면접이 종료된 경우에는
            기존 sessions API와 동일하게 결과를 저장하고 응답 payload에
            result_id를 포함한다. 리액션 발화를 이미 보낸 제출 건이면
            utterance_queue와 transcript의 리액션 문장을 제거해 음성과
            화면 양쪽의 중복 리액션을 막는다.

            Args:
                request:
                    coordinator가 제출 직전 검증한 불변 답변 snapshot.

            Returns:
                기존 sessions events API와 같은 JSON 직렬화 가능 세션 응답.
            """
            metrics_payload = (
                request.delivery_metrics.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                if request.delivery_metrics is not None
                else None
            )
            adapted_input = from_voice(
                request.session_id,
                request.question_id,
                {
                    "action": "submit",
                    "text": request.answer_text,
                    "metrics": metrics_payload,
                },
            )
            committed_state = await asyncio.to_thread(
                session.submit_event,
                adapted_input,
                client_event_id=request.client_event_id,
            )

            result_id = None
            if committed_state.finished:
                try:
                    result = _save_finished_result(
                        db=db,
                        current_user=current_user,
                        runtime_session_id=session_id,
                        session=session,
                    )
                    result_id = result.id
                except Exception:
                    logger.exception(
                        "자동 제출 후 면접 결과 저장에 실패했습니다.",
                        extra={"session_id": session_id},
                    )
            response = _session_response(
                committed_state,
                result_id=result_id,
            )
            if reaction_sent_event_id == request.client_event_id:
                _strip_predelivered_reaction(response)
            return response

        async def on_answer_committed(
            result: VoiceTurnCommitResult,
        ) -> None:
            """자동 제출 완료와 최신 세션 상태를 현재 WebSocket에 전달한다.

            메시지 전송 뒤에는 완료된 질문의 임시 buffer를 다음 질문용으로
            교체한다. 면접이 끝났으면 세션의 음성 턴 항목을 제거한다.

            Args:
                result:
                    제출에 사용한 snapshot과 최신 세션 응답.
            """
            nonlocal connection_open, coordinator
            if not owns_current_worker():
                return
            await send_model(
                AnswerCommittedMessage(
                    question_id=result.request.question_id,
                    revision=result.request.revision,
                    completion_reason=result.request.completion_reason,
                    session=result.session,
                )
            )

            next_question = result.session.get("question")
            if result.session.get("finished"):
                registry.remove(session_id)
                connection_open = False
                await websocket.close()
            elif isinstance(next_question, dict) and next_question.get("question_id"):
                next_entry = registry.replace_question(
                    session_id=session_id,
                    question_id=str(next_question["question_id"]),
                )
                next_coordinator = build_coordinator(next_entry)
                registry.attach_worker(
                    session_id=session_id,
                    worker=next_coordinator.worker,
                )
                coordinator = next_coordinator
                await next_coordinator.prepare_connection()

        async def on_commit_failed(
            request: VoiceTurnCommitRequest,
            reason: str,
        ) -> None:
            """자동 제출 실패를 복구 가능한 WebSocket 오류로 전달한다.

            Args:
                request:
                    실패한 자동 제출 snapshot.

                reason:
                    coordinator가 전달한 안정적인 오류 코드.
            """
            if not owns_current_worker():
                return
            await send_model(
                VoiceTurnErrorMessage(
                    code=reason,
                    recoverable=True,
                    message="음성 답변을 자동 제출하지 못했습니다.",
                )
            )

        def build_coordinator(
            entry: VoiceTurnRegistryEntry,
        ) -> VoiceTurnCoordinator:
            """현재 연결 callback을 사용하는 질문 단위 coordinator를 만든다.

            Args:
                entry:
                    현재 질문의 buffer와 lock을 보유한 registry 항목.

            Returns:
                현재 WebSocket 출력과 기존 제출 경로에 연결된 coordinator.
            """
            return VoiceTurnCoordinator(
                session=session,
                entry=entry,
                on_state_changed=on_state_changed,
                on_confirmation_requested=on_confirmation_requested,
                on_confirmation_cancelled=on_confirmation_cancelled,
                on_commit_started=on_commit_started,
                on_commit_answer=on_commit_answer,
                on_answer_committed=on_answer_committed,
                on_commit_failed=on_commit_failed,
            )

        coordinator = build_coordinator(entry)
        async with entry.lock:
            if entry.buffer.state == "committing":
                log_voice_turn_event(
                    "voice_turn.connection.rejected",
                    session_id=session_id,
                    question_id=entry.buffer.question_id,
                    revision=entry.buffer.revision,
                    answer_text=entry.buffer.answer_text,
                    reject_reason="turn_commit_in_progress",
                )
                await _send_fatal_error(
                    websocket=websocket,
                    send_model=send_model,
                    code="turn_commit_in_progress",
                    message=(
                        "음성 답변 제출이 진행 중입니다. 잠시 후 다시 연결해 주세요."
                    ),
                )
                return
            registry.attach_worker(
                session_id=session_id,
                worker=coordinator.worker,
            )
        await coordinator.prepare_connection()
        await send_model(
            ConnectionReadyMessage(
                session_id=session_id,
                question_id=entry.buffer.question_id,
                revision=entry.buffer.revision,
                state=entry.buffer.state,
            )
        )
        log_voice_turn_event(
            "voice_turn.connection.ready",
            session_id=session_id,
            question_id=entry.buffer.question_id,
            revision=entry.buffer.revision,
            answer_text=entry.buffer.answer_text,
            voice_turn_state=entry.buffer.state,
            reconnected=reconnecting,
        )

        while connection_open:
            try:
                raw_message = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:
                await send_model(
                    VoiceTurnErrorMessage(
                        code="invalid_message",
                        recoverable=True,
                        message="유효한 JSON 메시지를 보내 주세요.",
                    )
                )
                continue

            try:
                client_message = VOICE_TURN_CLIENT_MESSAGE_ADAPTER.validate_python(
                    raw_message
                )
            except ValidationError:
                await send_model(
                    VoiceTurnErrorMessage(
                        code="invalid_message",
                        recoverable=True,
                        message="음성 턴 메시지 형식을 확인해 주세요.",
                    )
                )
                continue

            try:
                if isinstance(client_message, AnswerTranscriptUpdatedMessage):
                    delivery_metrics = (
                        DeliveryMetrics.model_validate(
                            client_message.metrics.model_dump(
                                mode="json",
                                exclude_none=True,
                            )
                        )
                        if client_message.metrics is not None
                        else None
                    )
                    await coordinator.handle_transcript_updated(
                        question_id=client_message.question_id,
                        revision=client_message.revision,
                        text=client_message.text,
                        speech_active=client_message.speech_active,
                        segment_final=client_message.segment_final,
                        answer_duration_seconds=(
                            client_message.answer_duration_seconds
                        ),
                        delivery_metrics=delivery_metrics,
                    )
                elif isinstance(client_message, VoiceActivityChangedMessage):
                    buffer_snapshot = await coordinator.handle_activity_changed(
                        question_id=client_message.question_id,
                        revision=client_message.revision,
                        speech_active=client_message.speech_active,
                    )
                    if client_message.speech_active:
                        await send_model(
                            TurnStateChangedMessage(
                                question_id=buffer_snapshot.question_id,
                                revision=buffer_snapshot.revision,
                                reason="candidate_resumed_speaking",
                            )
                        )
                elif isinstance(
                    client_message,
                    TurnConfirmationResponseReadyMessage,
                ):
                    await coordinator.handle_confirmation_response_ready(
                        confirmation_id=client_message.confirmation_id,
                        question_id=client_message.question_id,
                        revision=client_message.revision,
                        playback_status=client_message.playback_status,
                    )
                elif isinstance(
                    client_message,
                    TurnConfirmationResponseActivityChangedMessage,
                ):
                    await coordinator.handle_confirmation_response_activity_changed(
                        confirmation_id=client_message.confirmation_id,
                        question_id=client_message.question_id,
                        revision=client_message.revision,
                        speech_active=client_message.speech_active,
                    )
                elif isinstance(
                    client_message,
                    TurnConfirmationRespondedMessage,
                ):
                    confirmation_result = (
                        await coordinator.handle_confirmation_response(
                            confirmation_id=client_message.confirmation_id,
                            question_id=client_message.question_id,
                            revision=client_message.revision,
                            response_revision=client_message.response_revision,
                            text=client_message.text,
                        )
                    )
                    if confirmation_result.decision.intent != "finish":
                        reason_by_intent = {
                            "continue": "candidate_wants_to_continue",
                            "answer_content": "additional_answer_content",
                            "unknown": "confirmation_unknown",
                        }
                        await send_model(
                            TurnStateChangedMessage(
                                question_id=(
                                    confirmation_result.buffer.question_id
                                ),
                                revision=confirmation_result.buffer.revision,
                                reason=reason_by_intent[
                                    confirmation_result.decision.intent
                                ],
                            )
                        )
            except VoiceTurnCoordinatorError as exc:
                await send_model(
                    VoiceTurnErrorMessage(
                        code=exc.code,
                        recoverable=exc.recoverable,
                        message=str(exc),
                    )
                )
                if not exc.recoverable:
                    await websocket.close(code=_POLICY_VIOLATION_CLOSE_CODE)
                    break
            except VoiceTurnQuestionMismatchError:
                await send_model(
                    VoiceTurnErrorMessage(
                        code="question_mismatch",
                        recoverable=True,
                        message="현재 질문과 다른 음성 이벤트입니다.",
                    )
                )
            except VoiceTurnAlreadyCommittedError:
                await send_model(
                    VoiceTurnErrorMessage(
                        code="turn_already_committed",
                        recoverable=True,
                        message="이미 제출된 음성 답변입니다.",
                    )
                )
            except VoiceTurnBufferError:
                await send_model(
                    VoiceTurnErrorMessage(
                        code="invalid_message",
                        recoverable=True,
                        message="음성 턴 상태와 이벤트를 확인해 주세요.",
                    )
                )
            except Exception:
                await send_model(
                    VoiceTurnErrorMessage(
                        code="internal_error",
                        recoverable=True,
                        message="음성 이벤트를 처리하지 못했습니다.",
                    )
                )
    except WebSocketDisconnect:
        pass
    finally:
        connection_open = False
        if coordinator is not None:
            log_voice_turn_event(
                "voice_turn.connection.closed",
                session_id=session_id,
                question_id=coordinator.worker.buffer.question_id,
                revision=coordinator.worker.buffer.revision,
                answer_text=coordinator.worker.buffer.answer_text,
                voice_turn_state=coordinator.worker.buffer.state,
            )
            await coordinator.aclose()
            registry.detach_worker(
                session_id=session_id,
                worker=coordinator.worker,
            )


def _strip_predelivered_reaction(response: dict) -> None:
    """선전송 리액션과 중복되는 면접관 리액션 문장을 세션 응답에서 제거한다.

    answer.reaction으로 리액션을 이미 전달한 제출 건에서는 utterance_queue의
    리액션 문장을 제거하고, transcript의 마지막 면접관 발화와 last_utterance를
    질문 본문으로 교체해 화면과 음성 모두에서 리액션이 두 번 나오지 않게 한다.
    off_topic이나 hint처럼 안내 문장 자체가 정보를 담는 턴과 질문이 없는 종료
    턴은 그대로 둔다. 그래프가 보관하는 세션 상태는 수정하지 않고 이 WebSocket
    응답 payload만 정리한다.

    Args:
        response:
            _session_response()가 만든 JSON 직렬화 가능 세션 응답. 제자리에서
            수정된다.
    """
    if response.get("turn_type") != "question":
        return

    utterance_queue = response.get("utterance_queue")
    if isinstance(utterance_queue, list) and len(utterance_queue) >= 2:
        response["utterance_queue"] = utterance_queue[1:]

    question = response.get("question")
    question_text = question.get("text") if isinstance(question, dict) else None
    if not question_text:
        return

    transcript = response.get("transcript")
    if isinstance(transcript, list) and transcript:
        last_turn = transcript[-1]
        if (
            isinstance(last_turn, dict)
            and last_turn.get("role") == "interviewer"
            and last_turn.get("text") != question_text
        ):
            last_turn["text"] = question_text

    if response.get("last_utterance"):
        response["last_utterance"] = question_text


async def _authenticate_first_message(
    *,
    websocket: WebSocket,
    db: Session,
    send_model: WebSocketModelSender,
) -> User | None:
    """연결 제한 시간 안에 첫 Access Token 인증 메시지를 검증한다.

    Args:
        websocket:
            첫 JSON 메시지를 받을 WebSocket 연결.

        db:
            토큰 사용자를 조회할 SQLAlchemy 세션.

        send_model:
            인증 오류를 직렬화해 보낼 WebSocket 전송 함수.

    Returns:
        인증된 User. 연결이 끊겼거나 인증에 실패하면 None.
    """
    try:
        raw_message = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=settings.voice_turn_auth_timeout_seconds,
        )
    except asyncio.TimeoutError:
        await _send_fatal_error(
            websocket=websocket,
            send_model=send_model,
            code="authentication_required",
            message="연결 인증 시간이 초과되었습니다.",
        )
        return None
    except WebSocketDisconnect:
        return None
    except Exception:
        await _send_fatal_error(
            websocket=websocket,
            send_model=send_model,
            code="authentication_required",
            message="첫 메시지로 연결 인증 정보가 필요합니다.",
        )
        return None

    try:
        auth_message = ConnectionAuthenticateMessage.model_validate(raw_message)
        return authenticate_access_token(auth_message.access_token, db)
    except (ValidationError, HTTPException):
        await _send_fatal_error(
            websocket=websocket,
            send_model=send_model,
            code="invalid_token",
            message="유효한 연결 인증 정보가 필요합니다.",
        )
        return None


async def _send_fatal_error(
    *,
    websocket: WebSocket,
    send_model: WebSocketModelSender,
    code: str,
    message: str,
) -> None:
    """복구할 수 없는 오류를 전송하고 WebSocket 연결을 종료한다.

    Args:
        websocket:
            종료할 WebSocket 연결.

        send_model:
            오류 모델을 JSON으로 전송할 함수.

        code:
            클라이언트가 인증·세션 오류를 구분할 안정적인 코드.

        message:
            내부 예외를 제외한 사용자 노출 가능 설명.
    """
    try:
        await send_model(
            VoiceTurnErrorMessage(
                code=code,
                recoverable=False,
                message=message,
            )
        )
    finally:
        await websocket.close(code=_POLICY_VIOLATION_CLOSE_CODE)
