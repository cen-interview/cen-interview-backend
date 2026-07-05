"""전체 흐름 배선 (4단계: 단순 함수 오케스트레이션).

세 에이전트(Strategy/Interviewer/Assessment)와 Evidence 툴을 하나로 연결한다.
이 파일이 "접착제"다 — 주인 없으면 아무도 안 짜는 영역이므로 Interviewer
담당(C)이 관리한다.

지금은 에이전트 내부 로직이 전부 스텁(가짜)이라 LangGraph 의 진짜 장점
(조건부 엣지 시각화, 체크포인터로 세션 영속화)을 쓸 이유가 아직 없다.
그래서 StateGraph 대신 얇은 파이썬 클래스로 "Strategy → Interviewer →
Assessment(→ Evidence)"를 연결만 해둔다.

흐름:
  create_session() ─▶ 첫 질문
  handle_event()   ─▶ (다음 질문 | None=종료)
  finalize()        ─▶ FinalReport

TODO(담당 C, 7~8단계): 음성 모드 재접속/장시간 세션이 필요해지면
  langgraph.graph.StateGraph(+ checkpointer) 로 교체. 그때도 아래
  create_session/get_session/InterviewSession 인터페이스는 유지해서
  API(api/main.py) 쪽 호출부가 바뀌지 않도록 한다.
"""

import uuid

from interview.assessment import AssessmentAgent
from interview.interviewer.agent import InterviewerAgent
from interview.interviewer.session import SessionState
from interview.schemas.events import InterviewerEvent, Mode
from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Question
from interview.schemas.report import FinalReport
from interview.strategy import StrategyAgent


class InterviewSession:
    """세션 1건에 필요한 에이전트 묶음 + 상태."""

    def __init__(
        self,
        session_id: str,
        mode: Mode,
        coverage: CoverageMap,
        max_questions: int = 10,
    ) -> None:
        self.strategy = StrategyAgent(coverage)
        self.assessment = AssessmentAgent()
        self.state = SessionState(
            session_id=session_id, mode=mode, max_questions=max_questions
        )
        self.interviewer = InterviewerAgent(self.state, self.strategy, self.assessment)

    def start(self) -> Question:
        """첫 질문을 뽑아 세션에 세팅하고 반환한다."""
        question = self.strategy.next_question(last_signal=None)
        self.state.current_question = question
        self.state.asked_count = 1
        return question

    def handle_event(self, event: InterviewerEvent) -> Question | None:
        """이벤트 1건 처리 → 다음 질문 (종료면 None)."""
        return self.interviewer.handle(event)

    def is_finished(self) -> bool:
        return self.state.finished

    def finalize(self) -> FinalReport:
        """면접 종료 후 최종 평가서 생성."""
        return self.assessment.finalize()


# [Stub] 세션 레지스트리. 영속화는 8단계에서 실제 DB/Redis 등으로 교체.
_sessions: dict[str, InterviewSession] = {}


def create_session(
    mode: Mode, coverage: CoverageMap | None = None, max_questions: int = 10
) -> tuple[InterviewSession, Question]:
    """세션 생성 + 첫 질문 반환. API 의 POST /sessions 가 호출."""
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    session = InterviewSession(session_id, mode, coverage or CoverageMap(), max_questions)
    first_question = session.start()
    _sessions[session_id] = session
    return session, first_question


def get_session(session_id: str) -> InterviewSession:
    """세션 조회. API 의 POST /events 가 호출.

    TODO(담당 C): 존재하지 않는 session_id 에 대한 에러 처리(404 매핑)는
    API 계층(5단계)에서.
    """
    return _sessions[session_id]
