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
from interview.interviewer.facade import create_session as create_interview_session, get_session
from interview.schemas.events import Mode
from uuid import uuid4

from interview.api.database import Base, engine
from interview.api.users.model import User
from interview.api.auth.model import RefreshToken
from interview.api.users.router import router as users_router
from interview.api.auth.router import router as auth_router
from interview.api.evidence.router import router as evidence_router

from interview.assessment import AssessmentAgent
from interview.interviewer.agent import InterviewerAgent
from interview.interviewer.session import SessionState
from interview.schemas.events import AnswerSubmitted, EndRequested
from interview.schemas.question import Difficulty, Question, QuestionKind
from interview.strategy.agent import StrategyAgent

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from interview.api.auth.dependency import get_current_user
from interview.api.database import Base, engine, get_db
from interview.api.interviews.service import save_interview_result

from dotenv import load_dotenv
from openai import OpenAI

from functools import lru_cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    #Base.metadata.create_all(bind=engine)
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

    session, first_question = create_interview_session(mode=mode)
    return {
        "session_id": session.state.session_id,
        "question": first_question.model_dump(),
    }


# ── 3. 면접 진행 (이벤트 수신) ────────────────────────────
class EventRequest(BaseModel):
    session_id: int
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

sessions: dict[str, SessionState] = {}
assessments: dict[str, AssessmentAgent] = {}


class AnswerRequest(BaseModel):
    text: str


@app.post("/api/sessions")
def create_session():
    session_id = str(uuid4())

    question = Question(
        question_id=str(uuid4()),
        text="FastAPI에서 Depends를 사용하는 이유는 무엇인가요?",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=QuestionKind.MAIN,
        evidence_ids=[],
    )

    session = SessionState(
        session_id=session_id,
        current_question=question,
        asked_count=1,
        max_questions=10,
        main_question_id=question.question_id,
        main_topic=question.topic,
        finished=False,
    )

    sessions[session_id] = session
    assessments[session_id] = AssessmentAgent()

    return {
        "session_id": session_id,
        "question": question,
        "finished": session.finished,
    }


@app.post("/api/sessions/{session_id}/answer")
def submit_answer(session_id: int, request: AnswerRequest):
    session = sessions.get(session_id)
    assessment = assessments.get(session_id)

    if session is None or assessment is None:
        raise HTTPException(status_code=404, detail="session not found")

    if session.finished:
        return {
            "session_id": session_id,
            "next_question": None,
            "finished": True,
        }

    if session.current_question is None:
        raise HTTPException(status_code=400, detail="current question not found")

    event = AnswerSubmitted(
        session_id=session_id,
        question_id=session.current_question.question_id,
        text=request.text,
    )

    interviewer = InterviewerAgent(
        session=session,
        strategy=StrategyAgent(),
        assessment=assessment,
    )

    next_question = interviewer.handle(event)

    return {
        "session_id": session_id,
        "next_question": next_question,
        "finished": session.finished,
    }


@app.post("/api/sessions/{session_id}/end")
def end_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = sessions.get(session_id)
    assessment = assessments.get(session_id)

    if session is None or assessment is None:
        raise HTTPException(
            status_code=404,
            detail="session not found",
        )

    event = EndRequested(session_id=session_id)

    interviewer = InterviewerAgent(
        session=session,
        strategy=StrategyAgent(),
        assessment=assessment,
    )

    interviewer.handle(event)

    # 최종 리포트 생성
    report = assessment.finalize()

    # 최종 리포트 DB 저장
    result = save_interview_result(
        db=db,
        user_id=current_user.id,
        session_id=session_id,
        report=report,
        topic_scores=assessment.competency.topic_scores,
    )

    return {
        "result_id": result.id,
        "session_id": session_id,
        "finished": session.finished,
        "report": report,
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
