"""Interviewer LangGraph를 감싸는 세션 단위 파사드와 인메모리 레지스트리."""

import uuid

from langgraph.types import Command

from interview.assessment import AssessmentAgent
from interview.interviewer.models import AdaptedInput, DeliveryMetrics
from interview.interviewer.session import SessionState
from interview.interviewer.workflow.graph import get_compiled_graph
from interview.interviewer.workflow.runtime import InterviewDeps
from interview.schemas.events import InterviewerEvent, Mode
from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Question
from interview.schemas.report import FinalReport
from interview.strategy import StrategyAgent


class InterviewSession:
    """세션별 의존성과 LangGraph 실행 설정을 감싸는 파사드.

    API나 입력 어댑터는 이 클래스만 사용하고, LangGraph의 체크포인터,
    thread_id, interrupt/resume 세부사항은 알지 않아도 된다. 실제 면접 흐름은
    항상 ``get_compiled_graph()``가 반환한 하나의 compiled graph에서 실행된다.

    Attributes:
        deps:
            현재 세션에서 사용할 Strategy와 Assessment 런타임 의존성.

        strategy:
            기존 호출부 호환을 위한 ``deps.strategy`` 편의 참조.

        assessment:
            기존 호출부 호환을 위한 ``deps.assessment`` 편의 참조.
    """

    def __init__(
        self,
        session_id: str,
        mode: Mode,
        coverage: CoverageMap,
        max_questions: int = 10,
        deps: InterviewDeps | None = None,
    ) -> None:
        """면접 세션의 초기 상태와 그래프 실행 컨텍스트를 준비한다.

        이 시점에는 그래프를 실행하지 않는다. 첫 질문 생성은
        ``start_session()``이 담당하므로 생성과 실행의 경계를 명확히 유지한다.

        Args:
            session_id:
                면접 세션을 식별하는 고유 ID.

            mode:
                면접 모드. ``chat`` 또는 ``voice``.

            coverage:
                Strategy가 질문 생성에 사용할 주제별 Evidence 커버리지.

            max_questions:
                세션에서 물어볼 최대 메인 질문 수.

            deps:
                테스트나 외부 조립 코드가 주입할 런타임 의존성. 없으면 실제
                StrategyAgent와 AssessmentAgent를 세션 전용으로 생성한다.
        """
        self.deps = deps or InterviewDeps(
            strategy=StrategyAgent(coverage),
            assessment=AssessmentAgent(),
        )
        self.strategy = self.deps.strategy
        self.assessment = self.deps.assessment
        self._initial_state = SessionState(
            session_id=session_id,
            mode=mode,
            max_questions=max_questions,
        )
        self._graph = get_compiled_graph()
        self._config = {"configurable": {"thread_id": session_id}}
        self._started = False

    @property
    def state(self) -> SessionState:
        """기존 ``session.state`` 호출부에 현재 체크포인트 상태를 제공한다."""
        if not self._started:
            return self._initial_state
        return self.get_state()

    def start_session(self) -> SessionState:
        """그래프를 시작하고 첫 질문이 준비된 세션 상태를 반환한다.

        최초 호출에서는 초기 SessionState를 그래프에 전달한다. 그래프는 greet와
        compose_utterance를 실행한 뒤 wait_event의 interrupt에서 멈춘다. 같은
        파사드에서 다시 호출하면 그래프를 중복 실행하지 않고 현재 상태를 반환한다.

        Returns:
            첫 질문과 면접관 발화가 채워진 현재 SessionState.
        """
        if self._started:
            return self.get_state()

        self._graph.invoke(
            self._initial_state,
            config=self._config,
            context=self.deps,
        )
        self._started = True
        return self.get_state()

    def submit_event(
        self,
        adapted_input: AdaptedInput | InterviewerEvent,
        delivery_metrics: DeliveryMetrics | None = None,
    ) -> SessionState:
        """정규화된 이벤트로 중단된 그래프를 재개한다.

        ``AdaptedInput``을 받으면 이벤트와 음성 전달 지표를 함께 사용한다. 기존
        호출부처럼 ``InterviewerEvent``만 전달하는 방식도 지원하며, 이 경우
        전달 지표는 선택 인자로 받을 수 있다. 종료된 세션에는 이벤트를 다시
        실행하지 않고 저장된 최종 상태를 그대로 반환한다.

        Args:
            adapted_input:
                입력 어댑터가 만든 AdaptedInput 또는 공통 InterviewerEvent.

            delivery_metrics:
                InterviewerEvent를 직접 전달할 때 사용할 선택적 음성 전달 지표.

        Returns:
            이벤트 처리가 끝나고 다음 interrupt 또는 END에 도달한 SessionState.

        Raises:
            RuntimeError:
                세션을 시작하기 전에 이벤트를 제출한 경우.
        """
        if not self._started:
            raise RuntimeError("session must be started before submitting an event")

        current_state = self.get_state()
        if current_state.finished:
            return current_state

        if isinstance(adapted_input, AdaptedInput):
            event = adapted_input.event
            metrics = adapted_input.delivery_metrics
        else:
            event = adapted_input
            metrics = delivery_metrics

        resume_payload = {
            "event": event.model_dump(mode="json"),
            "delivery_metrics": (
                metrics.model_dump(mode="json") if metrics is not None else None
            ),
        }
        self._graph.invoke(
            Command(resume=resume_payload),
            config=self._config,
            context=self.deps,
        )
        return self.get_state()

    def get_state(self) -> SessionState:
        """compiled graph의 최신 체크포인트를 SessionState로 복원한다.

        Returns:
            현재 thread_id에 저장된 최신 면접 세션 상태.

        Raises:
            RuntimeError:
                아직 그래프를 시작하지 않아 체크포인트가 없는 경우.
        """
        if not self._started:
            raise RuntimeError("session has not been started")

        snapshot = self._graph.get_state(self._config)
        return SessionState.model_validate(snapshot.values)

    def is_finished(self) -> bool:
        """현재 그래프 상태가 종료되었는지 반환한다."""
        return self._started and self.get_state().finished

    def start(self) -> Question:
        """기존 호출부 호환용으로 세션을 시작하고 첫 질문만 반환한다.

        Returns:
            그래프의 greet 노드가 생성한 첫 메인 질문.

        Raises:
            RuntimeError:
                그래프가 첫 질문을 생성하지 못한 경우.
        """
        state = self.start_session()
        if state.current_question is None:
            raise RuntimeError("graph did not create the first question")
        return state.current_question

    def handle_event(self, event: InterviewerEvent) -> Question | None:
        """기존 호출부 호환용으로 이벤트 처리 후 다음 질문만 반환한다.

        Args:
            event:
                처리할 공통 InterviewerEvent.

        Returns:
            계속 진행할 때 현재 질문, 세션이 종료되었으면 None.
        """
        state = self.submit_event(event)
        if state.finished:
            return None
        return state.current_question

    def finalize(self) -> FinalReport:
        """종료된 그래프 상태에 저장된 최종 리포트를 반환한다.

        Returns:
            그래프의 final_report 노드가 생성한 FinalReport.

        Raises:
            RuntimeError:
                세션이 끝나지 않았거나 최종 리포트가 저장되지 않은 경우.
        """
        state = self.get_state()
        if not state.finished or state.report is None:
            raise RuntimeError("final report is not available before session completion")
        return FinalReport.model_validate(state.report)


# 개발 단계용 인메모리 레지스트리. 프로세스 재시작과 멀티 프로세스 공유는
# 지원하지 않으며, 영속 레지스트리는 후속 단계에서 교체한다.
_sessions: dict[str, InterviewSession] = {}


def create_session(
    mode: Mode,
    coverage: CoverageMap | None = None,
    max_questions: int = 10,
) -> tuple[InterviewSession, Question]:
    """세션을 만들고 compiled graph가 생성한 첫 질문을 반환한다.

    API의 세션 생성 엔드포인트에서 호출하는 진입점이다. 아직 영속 저장소가
    없으므로 생성한 InterviewSession을 모듈 레벨 ``_sessions``에 보관한다.

    Args:
        mode:
            면접 모드. ``chat`` 또는 ``voice``.

        coverage:
            Evidence 파이프라인이 만든 주제별 근거 커버리지. 없으면 빈
            CoverageMap을 사용한다.

        max_questions:
            세션에서 물어볼 최대 메인 질문 수.

    Returns:
        생성된 InterviewSession과 첫 질문.
    """
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    session = InterviewSession(
        session_id=session_id,
        mode=mode,
        coverage=coverage or CoverageMap(),
        max_questions=max_questions,
    )
    first_question = session.start()
    _sessions[session_id] = session
    return session, first_question


def get_session(session_id: str) -> InterviewSession:
    """인메모리 레지스트리에서 면접 세션을 조회한다.

    Args:
        session_id:
            조회할 면접 세션 ID.

    Returns:
        해당 ID로 등록된 InterviewSession.

    Raises:
        KeyError:
            등록되지 않은 session_id인 경우.
    """
    return _sessions[session_id]
