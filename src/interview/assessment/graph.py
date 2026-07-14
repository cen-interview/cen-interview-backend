"""Assessment 평가 그래프.

답변 1개를 평가하는 내부 파이프라인의 state와 graph 노드를 정의한다.
"""

from uuid import uuid4

from pydantic import BaseModel, Field

from interview.assessment import evaluator
from interview.assessment.evaluator import JudgeResult
from interview.assessment.prompts import CONFLICT_CHECK_SYSTEM
from interview.assessment.scoring import AnswerAttempt
from interview.evidence.retrieval import search_evidence
from interview.llm.client import get_llm
from interview.schemas.evidence import EvidenceChunk
from interview.schemas.question import Question, QuestionCategory
from interview.schemas.signals import AnswerQualitySignal
from functools import lru_cache

class AssessmentState(BaseModel):
    """Assessment 내부 그래프에서 노드들이 공유하는 상태."""
    question: Question | None = None
    answer_text: str = ""
    delivery_metrics: dict | None = None
    history: list[AnswerAttempt] = Field(default_factory=list)
    user_id: str | None = None

    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    history_summary: str = ""
    judge_result: JudgeResult | None = None

    final_signal: AnswerQualitySignal | None = None

    same_topic_history: list[AnswerAttempt] = Field(default_factory=list)
    same_topic_history_summary: str = ""

def _normalize_topic(topic: str) -> str:
    return topic.strip().lower()


def _filter_same_topic_history(
    question: Question,
    history: list[AnswerAttempt],
) -> list[AnswerAttempt]:
    current_topic = _normalize_topic(question.topic)

    return [
        attempt
        for attempt in history
        if _normalize_topic(attempt.question_topic) == current_topic
    ]

def _build_same_topic_history_summary(history: list[AnswerAttempt]) -> str:
    if not history:
        return ""

    return "\n".join(
        (
            f"- question_id: {attempt.question_id}\n"
            f"  kind: {attempt.question_kind.value}\n"
            f"  question: {attempt.question_text}\n"
            f"  answer: {attempt.answer_text}\n"
            f"  quality: {attempt.signal.quality.value}\n"
            f"  rationale: {attempt.signal.rationale}"
        )
        for attempt in history
    )


def retrieve_evidence(state: AssessmentState) -> AssessmentState:
    """PROJECT 질문이면 Evidence를 조회하고, 그 외 질문은 근거 없이 진행한다."""

    question = state.question

    if question is None:
        return state

    if question.category != QuestionCategory.PROJECT:
        state.evidence_chunks = []
        return state

    state.evidence_chunks = search_evidence(
        query=f"{question.text}\n{state.answer_text}",
        topic=question.topic,
        user_id=state.user_id,
    )
    return state


def judge(state: AssessmentState) -> AssessmentState:
    """답변을 1차 판정하고 JudgeResult를 state에 저장한다."""

    if state.question is None:
        return state

    state.same_topic_history = _filter_same_topic_history(
        state.question,
        state.history,
    )

    state.same_topic_history_summary = _build_same_topic_history_summary(
    state.same_topic_history,
    )
    state.judge_result = evaluator._judge_with_llm(
        question=state.question,
        answer_text=state.answer_text,
        evidence_chunks=state.evidence_chunks,
        delivery_metrics=state.delivery_metrics,
        history=state.history,
    )

    return state


def conflict_check(state: AssessmentState) -> AssessmentState:
    """LLM으로 이전 답변/Evidence와의 충돌 여부를 정밀 확인한다."""

    if state.question is None or state.judge_result is None:
        return state

    if not state.same_topic_history_summary and not state.evidence_chunks:
        state.judge_result = state.judge_result.model_copy(
            update={"conflict_suspected": False}
        )
        return state

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(JudgeResult)
    conflict_result = structured_llm.invoke(
        [
            {"role": "system", "content": CONFLICT_CHECK_SYSTEM},
            {"role": "user", "content": _build_conflict_check_prompt(state)},
        ]
    )

    if conflict_result.conflict_suspected:
        state.judge_result = conflict_result.model_copy(
            update={
                "delivery_note": state.judge_result.delivery_note,
            }
        )
    else:
        state.judge_result = state.judge_result.model_copy(
            update={"conflict_suspected": False}
        )

    return state


def finalize_signal(state: AssessmentState) -> AssessmentState:
    """최종 JudgeResult를 Interviewer가 소비하는 AnswerQualitySignal로 변환한다."""

    if state.question is None or state.judge_result is None:
        return state

    state.final_signal = AnswerQualitySignal(
        answer_id=f"answer-{uuid4()}",
        question_id=state.question.question_id,
        quality=state.judge_result.quality,
        next_probe_target=state.judge_result.next_probe_target,
        conflict_type=state.judge_result.conflict_type,
        rationale=state.judge_result.rationale,
        accuracy=state.judge_result.accuracy,
        sufficiency=state.judge_result.sufficiency,
        delivery_note=state.judge_result.delivery_note,
    )

    return state


def route_after_judge(state: AssessmentState) -> str:
    """judge 결과에 따라 conflict_check 경유 여부를 결정한다."""

    if (
        state.judge_result is not None
        and state.judge_result.conflict_suspected
    ):
        return "conflict_check"

    return "finalize_signal"


def build_assessment_graph():
    """Assessment 내부 평가 그래프를 구성한다."""

    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AssessmentState)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("judge", judge)
    graph.add_node("conflict_check", conflict_check)
    graph.add_node("finalize_signal", finalize_signal)

    graph.add_edge(START, "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "judge")
    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "conflict_check": "conflict_check",
            "finalize_signal": "finalize_signal",
        },
    )
    graph.add_edge("conflict_check", "finalize_signal")
    graph.add_edge("finalize_signal", END)

    return graph

@lru_cache
def get_compiled_graph():
    """Assessment 내부 평가 그래프를 compile해서 반환한다."""

    return build_assessment_graph().compile()


def _build_conflict_check_prompt(state: AssessmentState) -> str:
    return f"""
[현재 질문]
question_id: {state.question.question_id}
topic: {state.question.topic}
category: {state.question.category}
kind: {state.question.kind}
difficulty: {state.question.difficulty}

질문:
{state.question.text}

[현재 답변]
{state.answer_text}

[같은 topic의 이전 답변 이력]
{state.same_topic_history_summary or "(없음)"}

[Evidence]
{evaluator._build_evidence_context(state.question, state.evidence_chunks)}

[1차 judge 결과]
quality: {state.judge_result.quality}
next_probe_target: {state.judge_result.next_probe_target}
rationale: {state.judge_result.rationale}
accuracy: {state.judge_result.accuracy}
sufficiency: {state.judge_result.sufficiency}

[판단 요청]
현재 답변이 이전 답변 또는 Evidence와 명확히 충돌하는지 판단하라.
충돌이 명확하면 quality=confirm_negative로 반환하라.
충돌이 명확하지 않으면 기존 1차 judge 결과를 유지할 수 있도록 conflict_suspected=false로 반환하라.
"""
