"""FastAPI 진입점.

실행:
    uv run uvicorn interview.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from interview.evidence import build_index
from interview.interviewer.adapters import from_chat, from_voice
from interview.interviewer.graph import create_session, get_session
from interview.schemas.events import Mode
from interview.api.database import Base, engine

# 모델 import: create_all이 테이블 정보를 알 수 있게 하기 위함
from interview.api.users.model import User
from interview.api.auth.model import RefreshToken

from interview.api.users.router import router as users_router
from interview.api.auth.router import router as auth_router

from dotenv import load_dotenv
from openai import OpenAI

from functools import lru_cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 SQLAlchemy 모델 기준으로 없는 테이블 자동 생성
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    title="Interview Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 회원 관련 API
app.include_router(users_router, prefix="/api")

# 인증 관련 API
app.include_router(auth_router, prefix="/api")

# 임시 ======================
load_dotenv()

@lru_cache
def get_openai_client() -> OpenAI:
    return OpenAI()

# ── 2. 면접 시작 + 모드 선택 ──────────────────────────────
class StartRequest(BaseModel):
    mode: str  # "voice" | "chat"


@app.post("/sessions")
def start_session(req: StartRequest):
    """세션 생성 + 첫 질문 선택."""
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"unknown mode: {req.mode}")

    session, first_question = create_session(mode=mode)
    return {
        "session_id": session.state.session_id,
        "question": first_question.model_dump(),
    }


# ── 3. 면접 진행 (이벤트 수신) ────────────────────────────
class EventRequest(BaseModel):
    session_id: str
    mode: str            # "voice" | "chat"
    payload: dict        # raw 입력 (어댑터가 해석)


@app.post("/events")
def post_event(req: EventRequest):
    """raw 입력을 공통 이벤트로 변환 후 세션에 투입, 다음 질문/종료를 반환."""
    try:
        session = get_session(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"session not found: {req.session_id}")

    try:
        event = (
            from_voice(req.session_id, req.payload)
            if req.mode == "voice"
            else from_chat(req.session_id, req.payload)
        )
        next_question = session.handle_event(event)
    except NotImplementedError:
        # 음성 모드 어댑터(from_voice)는 아직 미구현 (TODO 담당 C)
        raise HTTPException(status_code=501, detail="voice mode not implemented yet")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if session.is_finished():
        return {"finished": True, "report": session.finalize().model_dump()}

    return {
        "finished": False,
        "question": next_question.model_dump() if next_question else None,
    }

@app.post("/api/interview/realtime-transcription/token")
def create_realtime_transcription_token():
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
    return {"status": "ok"}
