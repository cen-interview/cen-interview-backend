"""기존 API가 사용하는 인메모리 InterviewSession 파사드와 세션 조회 함수."""

import uuid

from interview.assessment import AssessmentAgent
from interview.interviewer.agent import InterviewerAgent
from interview.interviewer.session import SessionState
from interview.interviewer.workflow.runtime import InterviewDeps
from interview.schemas.events import InterviewerEvent, Mode
from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Question
from interview.schemas.report import FinalReport
from interview.strategy import StrategyAgent

class InterviewSession:
    """세션 1건에 필요한 에이전트 묶음 + 상태.

    API 계층은 이 클래스의 인터페이스만 알면 된다. 내부가 나중에
    LangGraph로 바뀌더라도 create_session(), get_session(),
    handle_event() 형태는 유지하는 것이 목표다.

    Attributes:
        deps:
            Strategy/Assessment/LLM 같은 런타임 의존성 묶음.

        strategy:
            deps.strategy의 편의 참조. 첫 질문 생성과 다음 질문 생성에 사용한다.

        assessment:
            deps.assessment의 편의 참조. 답변 평가와 리포트 생성에 사용한다.

        state:
            session.py의 SessionState. 현재 질문, 질문 수, 종료 여부,
            이후 그래프 전환용 pending 값들을 보관한다.

        interviewer:
            실제 이벤트 라우팅을 담당하는 InterviewerAgent.
    """

    def __init__(
        self,
        session_id: str,
        mode: Mode,
        coverage: CoverageMap,
        max_questions: int = 10,
        deps: InterviewDeps | None = None,
    ) -> None:
        self.deps = deps or InterviewDeps(
            strategy=StrategyAgent(coverage),
            assessment=AssessmentAgent(),
        )
        self.strategy = self.deps.strategy
        self.assessment = self.deps.assessment
        self.state = SessionState(
            session_id=session_id, mode=mode, max_questions=max_questions
        )
        self.interviewer = InterviewerAgent(self.state, self.strategy, self.assessment)

    def start(self) -> Question:
        """첫 메인 질문을 생성하고 SessionState의 시작 필드를 세팅한다.

        세팅하는 값:
            - current_question: 첫 질문
            - asked_count: 1
            - main_question_id: 첫 메인 질문 ID
            - main_topic: 첫 메인 질문 주제
        """
        question = self.strategy.next_question(last_signal=None)
        self.state.current_question = question
        self.state.asked_count = 1
        self.state.main_question_id = question.question_id
        self.state.main_topic = question.topic
        return question

    def handle_event(self, event: InterviewerEvent) -> Question | None:
        """사용자 이벤트 1건을 처리한다.

        현재 구현에서는 이벤트별 세부 라우팅을 InterviewerAgent에 위임한다.
        반환값은 다음에 사용자에게 제시할 질문이며, 세션이 종료되면 None이다.
        """
        return self.interviewer.handle(event)

    def is_finished(self) -> bool:
        """세션 종료 여부를 반환한다."""
        return self.state.finished

    def finalize(self) -> FinalReport:
        """면접 종료 후 최종 평가서를 생성한다."""
        return self.assessment.finalize()


# [Stub] 세션 레지스트리. 영속화는 8단계에서 실제 DB/Redis 등으로 교체.
_sessions: dict[str, InterviewSession] = {}


def create_session(
    mode: Mode, coverage: CoverageMap | None = None, max_questions: int = 10
) -> tuple[InterviewSession, Question]:
    """세션을 만들고 첫 질문을 반환한다.

    API의 세션 생성 엔드포인트에서 호출하는 진입점이다. 아직 영속 저장소가
    없으므로 생성한 InterviewSession을 모듈 레벨 _sessions에 보관한다.

    Args:
        mode:
            면접 모드. "chat" 또는 "voice".

        coverage:
            Evidence 파이프라인이 만든 주제별 근거 커버리지. 없으면 빈
            CoverageMap을 사용한다.

        max_questions:
            세션에서 물어볼 최대 메인 질문 수.

    Returns:
        생성된 InterviewSession과 첫 질문.
    """
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    session = InterviewSession(session_id, mode, coverage or CoverageMap(), max_questions)
    first_question = session.start()
    _sessions[session_id] = session
    return session, first_question

def get_session(session_id: str) -> InterviewSession:
    """세션 조회. API 의 POST /events 가 호출.

    현재는 인메모리 dict에서 바로 꺼낸다. 서버 재시작, 멀티 프로세스,
    장시간 세션 복구는 아직 지원하지 않는다. 이후 LangGraph checkpointer나
    DB/Redis 기반 세션 저장소로 교체할 수 있다.

    TODO(담당 C): 존재하지 않는 session_id 에 대한 에러 처리(404 매핑)는
    API 계층(5단계)에서.
    """
    return _sessions[session_id]
