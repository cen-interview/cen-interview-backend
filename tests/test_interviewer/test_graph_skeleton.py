"""Interviewer LangGraph skeleton 흐름 테스트.

첫 실행에서 interrupt로 멈추고, 답변 resume을 반복했을 때 최대 메인 질문 수에
도달하는지 확인한다. Strategy/Assessment의 실제 구현 품질은 이 테스트의
관심사가 아니므로 최소 fake를 사용한다.
"""
# uv run pytest tests/test_interviewer/test_graph_skeleton.py -x
import uuid

from langgraph.types import Command

from interview.interviewer.session import SessionState
from interview.interviewer.workflow.graph import get_compiled_graph
from interview.interviewer.workflow.runtime import InterviewDeps
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)
from interview.schemas.report import FinalReport
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


def make_question(question_id: str = "q-1") -> Question:
    """그래프 skeleton 테스트에서 사용할 메인 질문을 만든다.

    Strategy fake가 반환하는 질문 fixture다. 그래프는 질문 객체의 ID와 topic을
    세션 상태에 옮기므로, 실제 질문 생성 로직 없이도 상태 전이를 검증할 수
    있도록 최소 필드만 채운다.

    Args:
        question_id:
            생성할 질문의 고유 ID.

    Returns:
        메인 질문 역할의 Question 모델.
    """
    return Question(
        question_id=question_id,
        text=f"{question_id} 테스트 질문입니다.",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=QuestionKind.MAIN,
        category=QuestionCategory.TECHNICAL,
    )


class FakeStrategy:
    """그래프가 Strategy에 기대하는 next_question 계약만 제공한다."""

    def __init__(self) -> None:
        """생성한 질문 수를 기록할 카운터를 초기화한다."""
        self.question_count = 0

    def next_question(self, last_signal: AnswerQualitySignal | None) -> Question:
        """다음 메인 질문을 순번 기반으로 반환한다.

        실제 StrategyAgent처럼 last_signal을 받을 수 있지만, skeleton 흐름에서는
        답변 품질별 분기를 검증하지 않으므로 값은 사용하지 않는다.

        Args:
            last_signal:
                직전 답변 평가 신호. 첫 질문에서는 None이다.

        Returns:
            순번이 포함된 question_id를 가진 메인 질문.
        """
        self.question_count += 1
        return make_question(question_id=f"q-{self.question_count}")


class FakeAssessment:
    """그래프가 Assessment에 기대하는 evaluate 계약만 제공한다."""

    def __init__(self) -> None:
        """평가 호출 내역을 보관할 리스트를 초기화한다."""
        self.evaluate_calls = []
        self.completed_sets = []

    def evaluate(
        self,
        question: Question,
        answer_text: str,
        delivery_metrics: dict | None = None,
    ) -> AnswerQualitySignal:
        """답변을 충분한 답변으로 고정 평가한다.

        LangGraph skeleton은 답변 평가 결과를 last_signal에 저장한 뒤 다음 메인
        질문 생성으로 넘긴다. 그래서 실제 채점 대신 항상 sufficient 신호를
        반환해 그래프의 반복 흐름만 검증한다.

        Args:
            question:
                현재 답변 대상 질문.

            answer_text:
                지원자가 제출한 답변 텍스트.

            delivery_metrics:
                음성 전달 지표. 채팅 테스트에서는 None이다.

        Returns:
            충분한 답변을 나타내는 AnswerQualitySignal.
        """
        self.evaluate_calls.append(
            {
                "question": question,
                "answer_text": answer_text,
                "delivery_metrics": delivery_metrics,
            }
        )
        return AnswerQualitySignal(
            answer_id=f"answer-{len(self.evaluate_calls)}",
            question_id=question.question_id,
            quality=AnswerQuality.SUFFICIENT,
            rationale=["fake assessment"],
        )

    def complete_question_set(self, main_question_id: str) -> None:
        """완료된 메인 질문 세트 ID를 기록한다.

        Args:
            main_question_id:
                평가가 완료된 질문 세트의 메인 질문 ID.
        """
        self.completed_sets.append(main_question_id)

    def finalize(self) -> FinalReport:
        """skeleton 종료 흐름에서 사용할 빈 최종 리포트를 반환한다.

        Returns:
            그래프 종료 상태 저장에 필요한 테스트용 FinalReport.
        """
        return FinalReport(
            summary="skeleton 테스트 완료",
            overall_score=0.0,
            strengths=[],
            improvement_points=[],
            learning_recommendations=[],
            evaluations=[],
        )


def initial_session_state() -> SessionState:
    """그래프 첫 invoke에 넘길 초기 세션 상태를 만든다.

    아직 질문이 생성되기 전 상태로 시작해야 greet 노드가 첫 질문을 만들고
    asked_count를 1로 세팅하는지 확인할 수 있다.

    Returns:
        max_questions가 10인 채팅 세션 상태.
    """
    return SessionState(session_id=f"session-{uuid.uuid4().hex}", max_questions=10)


def make_answer_payload(session_id: str, question_id: str) -> dict:
    """wait_event를 재개할 답변 제출 payload를 만든다.

    event dict는 events.py의 AnswerSubmitted 필드 구성을 따른다. 현재 skeleton
    그래프는 pending_event의 text만 평가에 사용하지만, 실제 이벤트 계약과
    어긋나지 않도록 type/session_id/question_id도 함께 둔다.

    Args:
        session_id:
            현재 그래프 상태의 세션 ID.

        question_id:
            현재 지원자에게 제시된 질문 ID.

    Returns:
        Command(resume=...)에 전달할 답변 payload.
    """
    return {
        "event": {
            "type": "answer_submitted",
            "session_id": session_id,
            "question_id": question_id,
            "text": "테스트 답변입니다.",
        },
        "delivery_metrics": None,
    }


def test_skeleton_reaches_end_after_ten_questions():
    """답변 resume을 반복하면 10번째 메인 질문에서 skeleton 그래프가 END에 도달한다."""
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    deps = InterviewDeps(strategy=FakeStrategy(), assessment=FakeAssessment())

    initial_state = initial_session_state()
    result = graph.invoke(initial_state, config=config, context=deps)
    assert "__interrupt__" in result
    assert result["asked_count"] == 1

    for _ in range(10):
        current_question = result["current_question"]
        result = graph.invoke(
            Command(
                resume=make_answer_payload(
                    initial_state.session_id,
                    current_question.question_id,
                )
            ),
            config=config,
            context=deps,
        )
        if result.get("finished"):
            break

    assert "__interrupt__" not in result
    assert result["finished"] is True
    assert result["asked_count"] == 10
    assert len(deps.assessment.evaluate_calls) == 10
