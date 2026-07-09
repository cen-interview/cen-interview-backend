import uuid
from dataclasses import dataclass
from typing import Any

from interview.assessment import AssessmentAgent
from interview.interviewer.agent import InterviewerAgent
from interview.interviewer.session import SessionState
from interview.schemas.events import InterviewerEvent, Mode
from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Question
from interview.schemas.report import FinalReport
from interview.schemas.signals import AnswerQualitySignal
from interview.strategy import StrategyAgent
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


@dataclass
class InterviewDeps:
    """세션 흐름에 필요한 런타임 의존성.

    Strategy/Assessment 같은 에이전트 인스턴스는 직렬화 대상 상태가 아니므로
    SessionState에 넣지 않는다. 현재는 InterviewSession이 직접 보관하고,
    이후 LangGraph 전환 시에는 runtime context로 그대로 옮길 수 있다.

    Attributes:
        strategy:
            다음 질문을 결정하고 생성하는 StrategyAgent.

        assessment:
            답변을 평가하고 최종 리포트를 만드는 AssessmentAgent.

        llm:
            이후 발화 레이어나 자연어 합성 단계에서 사용할 선택적 LLM client.
            지금 단계에서는 None이어도 전체 세션 흐름이 동작해야 한다.
    """

    strategy: StrategyAgent
    assessment: AssessmentAgent
    llm: object | None = None


def _state_get(state: SessionState | dict[str, Any], key: str, default: Any = None) -> Any:
    """SessionState 객체와 dict state 양쪽에서 값을 읽는다."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _serialize_signal(signal: AnswerQualitySignal) -> dict[str, Any]:
    """SessionState.last_signal에 저장할 수 있게 평가 신호를 dict로 바꾼다."""
    return signal.model_dump(mode="json")


def _restore_signal(raw_signal: Any) -> AnswerQualitySignal | None:
    """dict로 저장된 last_signal을 Strategy가 기대하는 모델로 복원한다."""
    if raw_signal is None:
        return None
    if isinstance(raw_signal, AnswerQualitySignal):
        return raw_signal
    return AnswerQualitySignal.model_validate(raw_signal)


def _runtime_deps(runtime: Any) -> InterviewDeps:
    """LangGraph runtime에서 InterviewDeps를 꺼낸다."""
    context = runtime.context
    if isinstance(context, InterviewDeps):
        return context
    return InterviewDeps(**context)


def greet(state: SessionState, runtime: Any) -> dict[str, Any]:
    """첫 메인 질문을 생성한다.

    3-2 skeleton의 시작 노드다. 첫 질문을 만들고 SessionState의 질문 진행
    필드를 초기화한다.
    """
    deps = _runtime_deps(runtime)
    question = deps.strategy.next_question(last_signal=None)

    return {
        "current_question": question,
        "asked_count": 1,
        "main_question_id": question.question_id,
        "main_topic": question.topic,
        "turn_type": "question",
        "finished": False,
        "error": None,
    }


def wait_event(state: SessionState, runtime: Any) -> dict[str, Any]:
    """지원자 입력을 기다리고 resume payload를 pending 필드에 저장한다.

    interrupt가 있는 노드는 재개 시 처음부터 다시 실행되므로, 여기에는
    다른 부작용을 두지 않는다.
    """
    payload = interrupt({"waiting_for": "candidate"})

    return {
        "pending_event": payload["event"],
        "pending_delivery_metrics": payload.get("delivery_metrics"),
    }


def evaluate_answer(state: SessionState, runtime: Any) -> dict[str, Any]:
    """현재 질문과 pending_event의 답변 텍스트를 평가한다.

    3단계 skeleton에서는 이벤트 타입 검증과 분기를 아직 하지 않는다.
    따라서 pending_event는 answer_submitted 형태라고 가정한다.
    """
    deps = _runtime_deps(runtime)
    current_question = _state_get(state, "current_question")
    pending_event = _state_get(state, "pending_event") or {}

    if current_question is None:
        return {"error": "current_question is missing", "finished": True}

    signal = deps.assessment.evaluate(
        question=current_question,
        answer_text=pending_event["text"],
        delivery_metrics=_state_get(state, "pending_delivery_metrics"),
    )

    return {
        "last_signal": _serialize_signal(signal),
        "error": None,
    }


def ask_main(state: SessionState, runtime: Any) -> dict[str, Any]:
    """답변 품질 분기 없이 다음 메인 질문을 생성한다.

    3-3 단계에서는 BONUS/MISCONCEPTION 같은 신호를 의도적으로 무시하고
    항상 다음 메인 질문으로 이동한다.
    """
    deps = _runtime_deps(runtime)
    asked_count = _state_get(state, "asked_count", 0)
    last_signal = _restore_signal(_state_get(state, "last_signal"))
    question = deps.strategy.next_question(last_signal=last_signal)

    return {
        "current_question": question,
        "asked_count": asked_count + 1,
        "main_question_id": question.question_id,
        "main_topic": question.topic,
        "challenge_used_in_set": False,
        "turn_type": "question",
        "pending_event": None,
        "pending_delivery_metrics": None,
    }


def after_ask(state: SessionState | dict[str, Any]) -> str:
    """ask_main 이후 계속 질문할지 종료할지 결정한다.

    라우팅 함수는 state를 읽기만 하고 변경하지 않는다.
    """
    finished = _state_get(state, "finished", False)
    asked_count = _state_get(state, "asked_count", 0)
    max_questions = _state_get(state, "max_questions", 10)
    return "end" if finished or asked_count >= max_questions else "continue"


builder = StateGraph(SessionState, context_schema=InterviewDeps)
builder.add_node("greet", greet)
builder.add_node("wait_event", wait_event)
builder.add_node("evaluate_answer", evaluate_answer)
builder.add_node("ask_main", ask_main)

builder.add_edge(START, "greet")
builder.add_edge("greet", "wait_event")
builder.add_edge("wait_event", "evaluate_answer")
builder.add_edge("evaluate_answer", "ask_main")
builder.add_conditional_edges(
    "ask_main",
    after_ask,
    {"end": END, "continue": "wait_event"},
)


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
