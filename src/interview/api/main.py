from uuid import uuid4
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from interview.assessment import AssessmentAgent
from interview.interviewer.agent import InterviewerAgent
from interview.interviewer.session import SessionState
from interview.schemas.events import AnswerSubmitted, EndRequested
from interview.schemas.question import Difficulty, Question, QuestionKind
from interview.strategy.agent import StrategyAgent

from interview.api.database import Base, engine
from interview.api.users.model import User
from interview.api.auth.model import RefreshToken
from interview.api.users.router import router as users_router
from interview.api.auth.router import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Interview Agent", lifespan=lifespan)

app.include_router(users_router)
app.include_router(auth_router)


sessions: dict[str, SessionState] = {}
assessments: dict[str, AssessmentAgent] = {}


class AnswerRequest(BaseModel):
    text: str


@app.post("/sessions")
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


@app.post("/sessions/{session_id}/answer")
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


@app.post("/sessions/{session_id}/end")
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


@app.get("/health")
def health():
    return {"status": "ok"}