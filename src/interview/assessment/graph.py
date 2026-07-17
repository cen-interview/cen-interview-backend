"""답변 하나를 평가하는 Assessment 내부 LangGraph 파이프라인.



프로젝트 질문은 Evidence를 조회해 평가하고, 기술 개념 질문은 LLM의
기술 지식을 기준으로 평가한다. 이전 같은 topic의 답변이 있거나
Evidence 충돌이 의심되는 경우에만 정밀 충돌 검사를 수행한다.

최종 결과는 Interviewer가 다음 질문 흐름을 결정할 수 있도록
AnswerQualitySignal 형태로 반환한다.
"""

from uuid import uuid4

from pydantic import BaseModel, Field

from interview.assessment import evaluator
from interview.assessment.evaluator import JudgeResult
from interview.assessment.scoring import AnswerAttempt
from interview.evidence.store import get_store

from interview.llm.logging import log_llm_output
from interview.schemas.evidence import EvidenceChunk
from interview.schemas.question import Question, QuestionCategory
from interview.schemas.signals import (
    AnswerQuality,
    AnswerQualitySignal,
    ConflictType,
)
from functools import lru_cache

import logging
logger = logging.getLogger("uvicorn.error")


class AssessmentState(BaseModel):
    """답변 1회 평가 과정에서 그래프 노드들이 공유하는 상태를 관리한다."""
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

# topic 비교를 위해 앞뒤 공백을 제거하고 소문자로 변환한다.
def _normalize_topic(topic: str) -> str:
    return topic.strip().lower()

# 전체 답변 이력에서 현재 질문과 같은 topic의 답변만 추출한다.
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

# 같은 topic의 이전 답변 이력을 충돌 검사 프롬프트 문자열로 변환한다.
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

def load_question_evidence(
    state: AssessmentState,
) -> AssessmentState:
    """질문 생성에 사용된 Evidence를 ID로 정확히 조회한다."""

    question = state.question

    if question is None:
        return state

    if question.category != QuestionCategory.PROJECT:
        state.evidence_chunks = []
        return state

    if not question.evidence_ids:
        state.evidence_chunks = []
        return state

    state.evidence_chunks = get_store().rank_chunks_by_query(
        query_text=question.text,
        chunk_ids=question.evidence_ids,
        k=3,
        user_id=state.user_id,
    )
    logger.info(
        "\n"
        "========== [PROJECT EVIDENCE LOADED] ==========\n"
        "question_id: %s\n"
        "question: %s\n"
        "requested_evidence_ids: %s\n"
        "loaded_evidence: %s\n"
        "===============================================\n",
        question.question_id,
        question.text,
        question.evidence_ids,
        [
            {
                "chunk_id": chunk.chunk_id,
                "topic": chunk.topic,
                "text": chunk.text,
            }
            for chunk in state.evidence_chunks
        ],
    )
    return state

# 답변을 LLM으로 1차 평가하고 JudgeResult를 상태에 저장한다.
def judge(state: AssessmentState) -> AssessmentState:

    if state.question is None:
        return state

    state.same_topic_history = _filter_same_topic_history(
        state.question,
        state.history,
    )

    state.same_topic_history_summary = _build_same_topic_history_summary(
    state.same_topic_history,
    )
    logger.info(
        "\n"
        "========== [ASSESSMENT LLM INPUT] ==========\n"
        "category: %s\n"
        "question_id: %s\n"
        "question: %s\n"
        "answer: %s\n"
        "evidence: %s\n"
        "============================================\n",
        state.question.category.value,
        state.question.question_id,
        state.question.text,
        state.answer_text,
        [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            }
            for chunk in state.evidence_chunks
        ],
    )
    state.judge_result = evaluator._judge_with_llm(
        question=state.question,
        answer_text=state.answer_text,
        evidence_chunks=state.evidence_chunks,
        delivery_metrics=state.delivery_metrics,
        history=state.history,
    )
    logger.info(
        "\n"
        "========== [ASSESSMENT LLM RESULT] ==========\n"
        "question_id: %s\n"
        "result: %s\n"
        "================================================\n",
        state.question.question_id,
        state.judge_result.model_dump(),
    )

    return state

def normalize_conflict(
    state: AssessmentState,
) -> AssessmentState:
    """Evidence 충돌이면 다른 quality를 무시하고 부정 확인으로 확정한다."""

    result = state.judge_result

    if result is None:
        return state

    if result.conflict_type == ConflictType.EVIDENCE_CONFLICT:
        state.judge_result = result.model_copy(
            update={
                "quality": AnswerQuality.CONFIRM_NEGATIVE,
                "conflict_suspected": False,
            }
        )

    return state


# 최종 JudgeResult를 Interviewer가 사용할 AnswerQualitySignal로 변환한다.
def finalize_signal(state: AssessmentState) -> AssessmentState:

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
    log_llm_output(
        "ANSWER_ASSESSMENT_FINAL",
        state.final_signal,
        metadata={
            "question_id": state.question.question_id,
            "topic": state.question.topic,
            "question_kind": state.question.kind.value,
            "conflict_check_applied": bool(
                state.same_topic_history_summary or state.evidence_chunks
            ),
        },
    )

    return state

def route_after_judge(
    state: AssessmentState,
) -> str:
    if (
        state.question is not None
        and state.question.category == QuestionCategory.PROJECT
    ):
        return "normalize_conflict"

    return "finalize_signal"

def route_by_category(
    state: AssessmentState,
) -> str:
    if (
        state.question is not None
        and state.question.category == QuestionCategory.PROJECT
    ):
        return "load_question_evidence"

    return "judge"
# 답변 평가 노드와 조건부 경로를 연결한 Assessment 그래프를 구성한다.
def build_assessment_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AssessmentState)

    graph.add_node(
        "load_question_evidence",
        load_question_evidence,
    )
    graph.add_node("judge", judge)
    graph.add_node("finalize_signal", finalize_signal)

    graph.add_conditional_edges(
        START,
        route_by_category,
        {
            "load_question_evidence": "load_question_evidence",
            "judge": "judge",
        },
    )

    graph.add_edge(
        "load_question_evidence",
        "judge",
    )
    graph.add_node(
        "normalize_conflict",
        normalize_conflict,
    )
    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "normalize_conflict": "normalize_conflict",
            "finalize_signal": "finalize_signal",
        },
    )
    graph.add_edge(
        "normalize_conflict",
        "finalize_signal",
    )
    graph.add_edge(
        "finalize_signal",
        END,
    )

    return graph

# 컴파일된 Assessment 그래프를 생성하고 프로세스에서 재사용한다.
@lru_cache
def get_compiled_graph():

    return build_assessment_graph().compile()


