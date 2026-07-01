"""FastAPI 진입점.

화면/클라이언트와 통신하는 얇은 계층. 비즈니스 로직은 전부 에이전트에 있고,
여기서는 (1) 인덱싱 트리거, (2) 면접 세션 시작, (3) 이벤트 수신 → 그래프 실행
정도만 한다. raw 입력은 interviewer.adapters 로 공통 이벤트로 변환한다.

실행:
    uv run uvicorn interview.api.main:app --reload
"""

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from interview.assessment import AssessmentAgent
from interview.evidence import build_index
from interview.evidence.store import get_store
from interview.interviewer.adapters import from_chat, from_voice
from interview.interviewer.agent import InterviewerAgent
from interview.interviewer.session import SessionState
from interview.schemas.events import AnswerSubmitted, EndRequested, InterviewerEvent
from interview.schemas.question import Question
from interview.schemas.report import FinalReport
from interview.strategy import StrategyAgent

app = FastAPI(title="Interview Agent")


@dataclass
class SessionRuntime:
    session: SessionState
    strategy: StrategyAgent
    assessment: AssessmentAgent
    interviewer: InterviewerAgent


SESSIONS: dict[str, SessionRuntime] = {}


# ── 1. 근거 자료 준비 (면접 시작 전, 1회) ─────────────────
class IndexRequest(BaseModel):
    notion_link: str
    github_links: list[str] = []


@app.post("/index")
def index(req: IndexRequest):
    """Notion/GitHub 를 인덱싱해 evidence_store 를 구축한다."""
    coverage = build_index(req.notion_link, req.github_links)
    return {"coverage": coverage.model_dump()}


# ── 2. 면접 시작 + 모드 선택 ──────────────────────────────
class StartRequest(BaseModel):
    mode: Literal["voice", "chat"]
    max_questions: int = 10


@app.post("/sessions")
def start_session(req: StartRequest):
    """세션 생성 + 첫 질문 선택.

    TODO(담당 C):
      - SessionState 생성, Strategy/Assessment/Interviewer 조립
      - graph.build_graph() 실행으로 첫 질문 획득
      - session_id + 첫 질문 반환
    """
    session_id = uuid4().hex
    coverage = get_store().build_coverage_map()
    session = SessionState(
        session_id=session_id,
        mode=req.mode,
        max_questions=req.max_questions,
    )
    strategy = StrategyAgent(coverage)
    assessment = AssessmentAgent()
    interviewer = InterviewerAgent(session, strategy, assessment)

    first_question = strategy.next_question(last_signal=None)
    session.current_question = first_question
    session.asked_count = 1

    SESSIONS[session_id] = SessionRuntime(
        session=session,
        strategy=strategy,
        assessment=assessment,
        interviewer=interviewer,
    )
    return _session_response(session_id, first_question)


# ── 3. 면접 진행 (이벤트 수신) ────────────────────────────
class EventRequest(BaseModel):
    session_id: str
    mode: Literal["voice", "chat"]
    payload: dict        # raw 입력 (어댑터가 해석)


@app.post("/events")
def post_event(req: EventRequest):
    """raw 입력을 공통 이벤트로 변환 후 그래프에 투입, 다음 질문/종료를 반환.

    TODO(담당 C): 세션 로드 → 어댑터 변환 → InterviewerAgent.handle → 응답
    """
    event = (
        from_voice(req.session_id, req.payload)
        if req.mode == "voice"
        else from_chat(req.session_id, req.payload)
    )
    return _handle_event(event)


class AnswerRequest(BaseModel):
    text: str
    question_id: str | None = None


@app.post("/sessions/{session_id}/answer")
def submit_answer(session_id: str, req: AnswerRequest):
    runtime = _get_runtime(session_id)
    question = runtime.session.current_question
    if question is None:
        raise HTTPException(status_code=400, detail="session has no active question")
    event = AnswerSubmitted(
        session_id=session_id,
        question_id=req.question_id or question.question_id,
        text=req.text,
    )
    return _handle_event(event)


@app.post("/sessions/{session_id}/end")
def end_session(session_id: str):
    return _handle_event(EndRequested(session_id=session_id))


@app.get("/health")
def health():
    return {"status": "ok"}


def _handle_event(event: InterviewerEvent) -> dict:
    runtime = _get_runtime(event.session_id)
    next_question = runtime.interviewer.handle(event)
    report = runtime.assessment.finalize() if runtime.session.finished else None
    return _session_response(event.session_id, next_question, report)


def _get_runtime(session_id: str) -> SessionRuntime:
    try:
        return SESSIONS[session_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc


def _session_response(
    session_id: str,
    question: Question | None,
    report: FinalReport | None = None,
) -> dict:
    runtime = _get_runtime(session_id)
    return {
        "session_id": session_id,
        "finished": runtime.session.finished,
        "question": question.model_dump() if question else None,
        "report": report.model_dump() if report else None,
    }
