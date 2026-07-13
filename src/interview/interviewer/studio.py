"""LangSmith Studio에서 Interviewer 그래프를 불러오는 개발용 진입점."""

from dataclasses import dataclass, field
from typing import Any

from interview.assessment import AssessmentAgent
from interview.interviewer.workflow.graph import _build_graph
from interview.interviewer.workflow.runtime import InterviewDeps
from interview.schemas.evidence import CoverageMap
from interview.strategy import StrategyAgent


def _create_strategy() -> StrategyAgent:
    """Studio 실행에 사용할 기본 질문 전략 에이전트를 생성한다.

    LangSmith Studio는 FastAPI의 InterviewSession 파사드를 거치지 않고 그래프를
    직접 실행한다. 따라서 별도의 사용자 Evidence가 없는 개발 환경에서도 첫
    질문을 생성할 수 있도록 빈 CoverageMap을 가진 StrategyAgent를 제공한다.

    Returns:
        빈 Evidence 커버리지로 초기화한 StrategyAgent.
    """
    return StrategyAgent(CoverageMap())


@dataclass
class StudioInterviewDeps(InterviewDeps):
    """Studio가 별도 객체 입력 없이 생성할 수 있는 Interviewer 의존성.

    Attributes:
        strategy:
            Studio의 각 runtime context에서 사용할 질문 전략 에이전트.

        assessment:
            후보자 답변을 평가하고 최종 리포트를 만드는 평가 에이전트.

        llm:
            발화 조립에서 선택적으로 사용할 LLM client. 기본값은 None이다.
    """

    strategy: Any = field(default_factory=_create_strategy)
    assessment: Any = field(default_factory=AssessmentAgent)
    llm: Any = None
    thread_id: str | None = None


# Agent Server가 checkpoint와 thread 상태를 관리하므로 checkpointer 없이 컴파일한다.
graph = _build_graph(context_schema=StudioInterviewDeps).compile()
