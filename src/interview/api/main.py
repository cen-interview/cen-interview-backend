from uuid import uuid4
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from interview.api.database import Base, engine
from interview.api.users.model import User
from interview.api.auth.model import RefreshToken
from interview.api.users.router import router as users_router
from interview.api.auth.router import router as auth_router

from interview.assessment import AssessmentAgent
from interview.interviewer.agent import InterviewerAgent
from interview.interviewer.session import SessionState
from interview.schemas.events import AnswerSubmitted, EndRequested
from interview.schemas.question import Difficulty, Question, QuestionKind
from interview.strategy.agent import StrategyAgent

from dotenv import load_dotenv
from openai import OpenAI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Base.metadata.create_all(bind=engine)
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


app.include_router(users_router, prefix="/api")
app.include_router(auth_router, prefix="/api")


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
def submit_answer(session_id: str, request: AnswerRequest):
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
def end_session(session_id: str):
    session = sessions.get(session_id)
    assessment = assessments.get(session_id)

    if session is None or assessment is None:
        raise HTTPException(status_code=404, detail="session not found")

    event = EndRequested(session_id=session_id)

    interviewer = InterviewerAgent(
        session=session,
        strategy=StrategyAgent(),
        assessment=assessment,
    )

    interviewer.handle(event)

    report = assessment.finalize()

    return {
        "session_id": session_id,
        "finished": session.finished,
        "report": report,
    }


load_dotenv()
client = OpenAI()


@app.post("/api/interview/realtime-transcription/token")
def create_realtime_transcription_token():
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