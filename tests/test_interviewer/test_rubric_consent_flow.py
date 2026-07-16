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
from interview.schemas.report import FinalReport, ReportGenerationResult
from interview.schemas.rubric import (
    RubricCandidate,
    RubricCriterion,
    RubricSource,
)
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


class FakeStrategy:
    def next_question(self, last_signal):
        return Question(
            question_id="q-technical",
            text="FastAPI 비동기 함수는 언제 사용하나요?",
            topic="FastAPI",
            difficulty=Difficulty.MEDIUM,
            kind=QuestionKind.MAIN,
            category=QuestionCategory.TECHNICAL,
        )


class FakeAssessment:
    def __init__(self) -> None:
        self.report_sources: list[RubricSource] | None = None

    def evaluate(self, question, answer_text, delivery_metrics=None):
        return AnswerQualitySignal(
            answer_id="answer-1",
            question_id=question.question_id,
            quality=AnswerQuality.SUFFICIENT,
        )

    def complete_question_set(self, main_question_id):
        return None

    def collect_rubric_sources(self):
        return [
            RubricSource(
                question_id="q-technical",
                topic="FastAPI",
                question="FastAPI 비동기 함수는 언제 사용하나요?",
                answer="비동기 I/O에서 사용합니다.",
            )
        ]

    def finalize_with_rubrics(self, rubric_sources=None):
        self.report_sources = list(rubric_sources or [])
        candidates = []
        if self.report_sources:
            candidates = [
                RubricCandidate(
                    question_id="q-technical",
                    topic="FastAPI",
                    question="FastAPI 비동기 함수는 언제 사용하나요?",
                    criteria=[
                        RubricCriterion(
                            criterion_id="async-io",
                            description="비동기 I/O를 설명한다.",
                        )
                    ],
                )
            ]
        return ReportGenerationResult(
            report=FinalReport(
                summary="완료",
                overall_score=80,
                strengths=[],
                improvement_points=[],
                learning_recommendations=[],
                evaluations=[],
            ),
            rubric_candidates=candidates,
        )


def _reach_consent():
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    assessment = FakeAssessment()
    deps = InterviewDeps(strategy=FakeStrategy(), assessment=assessment)
    session_id = f"session-{uuid.uuid4().hex}"
    state = graph.invoke(
        SessionState(session_id=session_id, max_questions=1),
        config=config,
        context=deps,
    )
    state = graph.invoke(
        Command(resume={
            "event": {
                "type": "answer_submitted",
                "session_id": session_id,
                "question_id": "q-technical",
                "text": "비동기 I/O에서 사용합니다.",
            },
            "delivery_metrics": None,
        }),
        config=config,
        context=deps,
    )
    return graph, config, deps, assessment, state


def test_consent_is_requested_before_final_report_and_rejection_skips_rubric():
    graph, config, deps, assessment, state = _reach_consent()

    assert "__interrupt__" in state
    assert state["rubric_share_status"] == "pending"
    assert state.get("report") is None
    assert assessment.report_sources is None

    finished = graph.invoke(
        Command(resume={"share": False}),
        config=config,
        context=deps,
    )

    assert finished["finished"] is True
    assert finished["rubric_share_status"] == "discarded"
    assert assessment.report_sources == []


def test_approval_generates_rubric_in_report_call_and_then_saves(monkeypatch):
    saved: list[RubricCandidate] = []

    class FakeStore:
        def add_candidate(self, candidate):
            saved.append(candidate)

    monkeypatch.setattr(
        "interview.interviewer.workflow.nodes.get_rubric_store",
        lambda: FakeStore(),
    )
    graph, config, deps, assessment, _ = _reach_consent()

    finished = graph.invoke(
        Command(resume={"share": True}),
        config=config,
        context=deps,
    )

    assert finished["finished"] is True
    assert finished["rubric_share_status"] == "shared"
    assert [source.question_id for source in assessment.report_sources] == [
        "q-technical"
    ]
    assert [candidate.question_id for candidate in saved] == ["q-technical"]
