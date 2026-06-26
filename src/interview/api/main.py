"""FastAPI 진입점.

화면/클라이언트와 통신하는 얇은 계층. 비즈니스 로직은 전부 에이전트에 있고,
여기서는 (1) 인덱싱 트리거, (2) 면접 세션 시작, (3) 이벤트 수신 → 그래프 실행
정도만 한다. raw 입력은 interviewer.adapters 로 공통 이벤트로 변환한다.

실행:
    uv run uvicorn interview.api.main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from interview.evidence import build_index
from interview.interviewer.adapters import from_chat, from_voice

app = FastAPI(title="Interview Agent")


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
    """세션 생성 + 첫 질문 선택.

    TODO(담당 C):
      - SessionState 생성, Strategy/Assessment/Interviewer 조립
      - graph.build_graph() 실행으로 첫 질문 획득
      - session_id + 첫 질문 반환
    """
    raise NotImplementedError


# ── 3. 면접 진행 (이벤트 수신) ────────────────────────────
class EventRequest(BaseModel):
    session_id: str
    mode: str            # "voice" | "chat"
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
    _ = event
    raise NotImplementedError


@app.get("/health")
def health():
    return {"status": "ok"}
