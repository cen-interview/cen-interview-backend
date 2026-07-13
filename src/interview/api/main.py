"""FastAPI 진입점.

실행:
    uv run uvicorn interview.api.main:app --reload
"""

from contextlib import asynccontextmanager
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from interview.api.auth.model import RefreshToken
from interview.api.auth.router import router as auth_router
from interview.api.database import Base, engine
from interview.api.evidence.router import router as evidence_router
from interview.api.users.model import User
from interview.api.users.router import router as users_router
from interview.interviewer.adapters import from_chat, from_voice
from interview.interviewer.facade import (
    create_session as create_interview_session,
    get_session,
)
from interview.interviewer.session import SessionState
from interview.schemas.events import Mode


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작과 종료 수명주기를 관리한다.

    Args:
        app:
            수명주기를 적용할 FastAPI 애플리케이션.
    """
    # Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Interview Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://cen-interview-front.vercel.app",
        "https://cen-interview.site",
        "https://www.cen-interview.site",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(evidence_router, prefix="/api")

load_dotenv()


@lru_cache
def get_openai_client() -> OpenAI:
    """환경 설정을 사용하는 OpenAI client를 프로세스에서 재사용한다."""
    return OpenAI()


class StartRequest(BaseModel):
    """면접 세션 생성 요청.

    Attributes:
        mode:
            면접 진행 모드. ``chat`` 또는 ``voice``.
    """

    mode: str


@app.post("/api/sessions")
def start_session(req: StartRequest):
    """면접 세션을 생성하고 compiled graph가 만든 첫 질문을 반환한다.

    Args:
        req:
            면접 시작 요청. mode는 ``chat`` 또는 ``voice``여야 한다.

    Returns:
        생성된 세션 ID와 첫 질문, 면접관 발화 큐, 종료 여부.

    Raises:
        HTTPException:
            지원하지 않는 mode가 전달되면 400 응답을 반환한다.
    """
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"unknown mode: {req.mode}")

    session, _ = create_interview_session(mode=mode)
    return _session_response(session.get_state())


class EventRequest(BaseModel):
    """세션에 전달할 채팅 또는 음성 raw 이벤트 요청.

    Attributes:
        payload:
            입력 어댑터가 해석할 action과 관련 데이터를 담은 dict. session_id는
            URL 경로에서 받고 mode는 서버에 저장된 세션 상태를 사용한다.
    """

    payload: dict


@app.post("/api/sessions/{session_id}/events")
def post_event(session_id: str, req: EventRequest):
    """raw 입력을 세션 mode에 맞게 변환하고 중단된 그래프를 재개한다.

    세션 생성 시 저장한 mode와 현재 질문 ID를 사용해 AdaptedInput을 만든다.
    종료된 세션이면 그래프를 다시 실행하지 않고 기존 상태와 리포트를 반환한다.

    Args:
        session_id:
            이벤트를 전달할 면접 세션 ID.

        req:
            action과 채널별 입력 데이터가 담긴 이벤트 요청.

    Returns:
        이벤트 처리 후 최신 질문, 발화 큐, 종료 여부와 선택적 리포트.

    Raises:
        HTTPException:
            세션이 없으면 404, payload를 변환할 수 없으면 400을 반환한다.
    """
    try:
        session = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")

    state = session.get_state()
    if state.finished:
        return _session_response(state)

    question_id = (
        state.current_question.question_id
        if state.current_question is not None
        else ""
    )

    try:
        adapted_input = (
            from_voice(session_id, question_id, req.payload)
            if state.mode == Mode.VOICE.value
            else from_chat(session_id, question_id, req.payload)
        )
        state = session.submit_event(adapted_input)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _session_response(state)


def _session_response(state: SessionState) -> dict:
    """SessionState를 세션 생성과 이벤트 API의 공통 응답으로 변환한다.

    Args:
        state:
            compiled graph의 최신 체크포인트에서 복원한 세션 상태.

    Returns:
        세션 ID, 현재 질문, 면접관 발화 큐, 오류, 종료 여부와 최종 리포트를
        JSON 직렬화 가능한 값으로 정리한 dict.
    """
    return {
        "session_id": state.session_id,
        "finished": state.finished,
        "question": (
            state.current_question.model_dump(mode="json")
            if state.current_question is not None and not state.finished
            else None
        ),
        "utterance_queue": state.utterance_queue,
        "error": state.error,
        "report": state.report if state.finished else None,
    }


@app.post("/api/interview/realtime-transcription/token")
def create_realtime_transcription_token():
    """OpenAI Realtime 전사용 단기 client secret을 발급한다."""
    client = get_openai_client()

    token = client.realtime.client_secrets.create(
        expires_after={
            "anchor": "created_at",
            "seconds": 60,
        },
        session={
            "type": "transcription",
            "audio": {
                "input": {
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                        "language": "ko",
                        "delay": "high",
                    },
                    "turn_detection": None,
                },
            },
        },
    )

    return {
        "value": token.value,
        "expires_at": token.expires_at,
    }


@app.get("/api/health")
def health():
    """서버 프로세스의 기본 상태를 반환한다."""
    return {"status": "ok"}
