"""실시간 음성 전사문과 턴 상태를 연결하는 WebSocket 라우터."""

import asyncio
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from interview.api.auth.dependency import authenticate_access_token
from interview.api.database import get_db
from interview.api.interviews.service import get_interview_session_by_runtime_id
from interview.api.users.model import User
from interview.api.voice.schema import (
    VOICE_TURN_CLIENT_MESSAGE_ADAPTER,
    AnswerTranscriptUpdatedMessage,
    ConnectionAuthenticateMessage,
    ConnectionReadyMessage,
    TurnStateChangedMessage,
    VoiceActivityChangedMessage,
    VoiceTurnErrorMessage,
)
from interview.config import settings
from interview.interviewer.facade import get_session
from interview.interviewer.models import DeliveryMetrics
from interview.interviewer.turn_completion.buffer import (
    VoiceTurnAlreadyCommittedError,
    VoiceTurnBufferError,
    VoiceTurnQuestionMismatchError,
)
from interview.interviewer.turn_completion.coordinator import (
    VoiceTurnCoordinator,
    VoiceTurnCoordinatorError,
)
from interview.interviewer.turn_completion.models import TurnCompletionResult
from interview.interviewer.turn_completion.registry import get_voice_turn_registry
from interview.schemas.events import Mode


router = APIRouter()

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

        async def on_result(result: TurnCompletionResult) -> None:
            """최신 keep_listening 판단을 현재 WebSocket에 전달한다.

            Args:
                result:
                    현재 buffer에 실제 기록된 최신 완료 판단 결과.
            """
            if not connection_open or coordinator is None:
                return
            try:
                current_entry = registry.get(session_id)
            except KeyError:
                return
            if current_entry.worker is not coordinator.worker:
                return
            if result.decision.recommended_action != "keep_listening":
                return
            try:
                await send_model(
                    TurnStateChangedMessage(
                        question_id=result.question_id,
                        revision=result.revision,
                        reason=result.decision.reason_code,
                    )
                )
            except Exception:
                return

        coordinator = VoiceTurnCoordinator(
            session=session,
            entry=entry,
            on_result=on_result,
        )
        registry.attach_worker(
            session_id=session_id,
            worker=coordinator.worker,
        )
        await send_model(
            ConnectionReadyMessage(
                session_id=session_id,
                question_id=entry.buffer.question_id,
                revision=entry.buffer.revision,
                state=entry.buffer.state,
            )
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
            await coordinator.aclose()
            registry.detach_worker(
                session_id=session_id,
                worker=coordinator.worker,
            )


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
