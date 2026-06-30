"""FastAPI 진입점.

화면/클라이언트와 통신하는 얇은 계층. 비즈니스 로직은 전부 에이전트에 있고,
여기서는 (1) 인덱싱 트리거, (2) 면접 세션 시작, (3) 이벤트 수신 → 그래프 실행
정도만 한다. raw 입력은 interviewer.adapters 로 공통 이벤트로 변환한다.

실행:
    uv run uvicorn interview.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from interview.evidence import build_index
from interview.interviewer.adapters import from_chat, from_voice
from interview.interviewer.graph import create_session, get_session
from interview.schemas.events import Mode

app = FastAPI(title="Interview Agent")

app.add_middleware( # 프론트엔드 연동용
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/health")
def health():
    return {"status": "ok"}
