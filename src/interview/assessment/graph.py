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
from interview.assessment.prompts import CONFLICT_CHECK_SYSTEM
from interview.assessment.scoring import AnswerAttempt
from interview.evidence.retrieval import search_evidence
from interview.evidence.store import get_store
from interview.llm.client import get_llm
from interview.llm.logging import log_llm_error, log_llm_output
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
ASSESSMENT_EVIDENCE_TOP_K = 5
ASSESSMENT_EVIDENCE_CONFIDENCE_THRESHOLD = 0.3

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

# 프로젝트 질문이면 관련 Evidence를 조회해 평가 상태에 저장한다.
# def retrieve_evidence(
#     state: AssessmentState,
# ) -> AssessmentState:
#     question = state.question

#     if question is None:
#         return state

#     if question.category != QuestionCategory.PROJECT:
#         state.evidence_chunks = []
#         return state

#     search_query = (
#         f"[질문]\n{question.text}\n\n"
#         f"[지원자 답변]\n{state.answer_text}"
#     )
#     logger.info(
#     "\n"
#     "========== [CHUNK SEARCH STEP 1] ==========\n"
#     "user_id: %s\n"
#     "question_id: %s\n"
#     "category: %s\n"
#     "topic: %s\n"
#     "evidence_ids: %s\n"
#     "question: %s\n"
#     "answer: %s\n"
#     "search_query:\n%s\n"
#     "===========================================",
#     state.user_id,
#     question.question_id,
#     question.category.value,
#     question.topic,
#     question.evidence_ids,
#     question.text,
#     state.answer_text,
#     search_query,
# )


#     evidence_chunks = search_evidence(
#         query=search_query,
#         topic=question.topic,
#         k=ASSESSMENT_EVIDENCE_TOP_K,
#         user_id=state.user_id,
#     )
#     question_evidence_ids = set(question.evidence_ids)
#     retrieved_evidence_ids = {
#     chunk.chunk_id
#     for chunk in evidence_chunks
# }
#     reliable_chunks = [
#         chunk
#         for chunk in evidence_chunks
#         if chunk.confidence
#         >= ASSESSMENT_EVIDENCE_CONFIDENCE_THRESHOLD
#     ]

#     logger.info(
#     "\n"
#     "========== [CHUNK SEARCH STEP 2 - RESULT] ==========\n"
#     "question evidence_ids: %s\n"
#     "retrieved chunk_ids: %s\n"
#     "overlap: %s\n"
#     "missing from retrieval: %s\n"
#     "extra retrieval chunks: %s\n"
#     "====================================================",
#     question.evidence_ids,
#     [chunk.chunk_id for chunk in evidence_chunks],
#     sorted(question_evidence_ids & retrieved_evidence_ids),
#     sorted(question_evidence_ids - retrieved_evidence_ids),
#     sorted(retrieved_evidence_ids - question_evidence_ids),
# )
    

#     state.evidence_chunks = reliable_chunks
#     return state

ANSWER_EVIDENCE_SIMILARITY_THRESHOLD = 0.3


def evidence_similarity_check(
    state: AssessmentState,
) -> AssessmentState:
    if state.question is None:
        return state

    if state.question.category != QuestionCategory.PROJECT:
        return state

    if not state.question.evidence_ids:
        return state

    results = get_store().score_answer_against_chunks(
        answer_text=state.answer_text,
        chunk_ids=state.question.evidence_ids,
        user_id=state.user_id,
    )

    state.evidence_chunks = [
        result.chunk
        for result in results
    ]

    if not results:
        return state

    max_similarity = max(
        result.score
        for result in results
    )
    logger.info(
    "[ANSWER_EVIDENCE_SIMILARITY] "
    "question_id=%s max_similarity=%.4f scores=%s",
    state.question.question_id,
    max_similarity,
    [
        {
            "chunk_id": result.chunk.chunk_id,
            "score": round(result.score, 4),
        }
        for result in results
    ],
)

    if max_similarity < ANSWER_EVIDENCE_SIMILARITY_THRESHOLD:
        state.judge_result = JudgeResult(
            quality=AnswerQuality.CONFIRM_NEGATIVE,
            next_probe_target=state.question.topic,
            rationale=[
                f"답변과 질문 근거 Evidence의 최대 유사도가 "
                f"{max_similarity:.3f}으로 기준보다 낮아 "
                "프로젝트 구현 사실을 다시 확인해야 합니다."
            ],
            conflict_type=ConflictType.EVIDENCE_CONFLICT,
            conflict_suspected=True,
            accuracy=max(0.0, max_similarity),
            sufficiency=max(0.0, max_similarity),
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
    state.judge_result = evaluator._judge_with_llm(
        question=state.question,
        answer_text=state.answer_text,
        evidence_chunks=state.evidence_chunks,
        delivery_metrics=state.delivery_metrics,
        history=state.history,
    )

    return state

# 이전 답변과 Evidence를 기준으로 현재 답변의 충돌 여부를 정밀 검사한다.
# def conflict_check(state: AssessmentState) -> AssessmentState:

#     if state.question is None or state.judge_result is None:
#         return state
#     if (
#         state.question.category == QuestionCategory.PROJECT
#         and state.question.evidence_ids
#     ):
#         similarity_results = (
#             get_store().score_answer_against_chunks(
#                 answer_text=state.answer_text,
#                 chunk_ids=state.question.evidence_ids,
#                 user_id=state.user_id,
#             )
#         )

#         logger.info(
#             "\n"
#             "========== [ANSWER EVIDENCE SIMILARITY] ==========\n"
#             "question_id: %s\n"
#             "answer: %s\n"
#             "scores: %s\n"
#             "max_similarity: %s\n"
#             "==================================================",
#             state.question.question_id,
#             state.answer_text,
#             [
#                 {
#                     "chunk_id": result.chunk.chunk_id,
#                     "score": round(result.score, 4),
#                     "text": result.chunk.text[:300],
#                 }
#                 for result in similarity_results
#             ],
#             (
#                 round(max(result.score for result in similarity_results), 4)
#                 if similarity_results
#                 else None
#             ),
#         )    

#     if not state.same_topic_history_summary and not state.evidence_chunks:
#         state.judge_result = state.judge_result.model_copy(
#             update={"conflict_suspected": False}
#         )
#         return state

#     llm = get_llm(temperature=0.0)
#     structured_llm = llm.with_structured_output(JudgeResult)
#     conflict_prompt = _build_conflict_check_prompt(state)
#     try:
#         conflict_result = structured_llm.invoke(
#             [
#                 {"role": "system", "content": CONFLICT_CHECK_SYSTEM},
#                 {"role": "user", "content": conflict_prompt},
#             ]
#         )
#         log_llm_output(
#             "CONFLICT_ASSESSMENT",
#             conflict_result,
#             metadata={
#                 "question_id": state.question.question_id,
#                 "topic": state.question.topic,
#                 "question_kind": state.question.kind.value,
#                 "evidence_ids": [chunk.chunk_id for chunk in state.evidence_chunks],
#                 "same_topic_history_count": len(state.same_topic_history),
#             },
#             input_data={
#                 "answer_text": state.answer_text,
#                 "conflict_prompt": conflict_prompt,
#             },
#         )
#     except Exception as exc:
#         log_llm_error(
#             "CONFLICT_ASSESSMENT",
#             exc,
#             metadata={
#                 "question_id": state.question.question_id,
#                 "topic": state.question.topic,
#                 "question_kind": state.question.kind.value,
#                 "evidence_ids": [chunk.chunk_id for chunk in state.evidence_chunks],
#                 "same_topic_history_count": len(state.same_topic_history),
#             },
#             input_data={
#                 "answer_text": state.answer_text,
#                 "conflict_prompt": conflict_prompt,
#             },
#         )
#         raise

#     if conflict_result.conflict_suspected:
#         state.judge_result = conflict_result.model_copy(
#             update={
#                 "delivery_note": state.judge_result.delivery_note,
#             }
#         )
#     else:
#         state.judge_result = state.judge_result.model_copy(
#             update={"conflict_suspected": False}
#         )

#     return state

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

# 1차 평가 결과에 따라 충돌 검사 노드를 실행할지 결정한다.
def route_after_judge(state: AssessmentState) -> str:

    if (
        state.judge_result is not None
        and state.judge_result.conflict_suspected
    ):
        return "conflict_check"

    return "finalize_signal"
def route_after_similarity(
    state: AssessmentState,
) -> str:
    if (
        state.judge_result is not None
        and state.judge_result.quality
        == AnswerQuality.CONFIRM_NEGATIVE
    ):
        return "finalize_signal"

    return "judge"
# 답변 평가 노드와 조건부 경로를 연결한 Assessment 그래프를 구성한다.
def build_assessment_graph():

    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AssessmentState)
    graph.add_node("evidence_similarity_check",evidence_similarity_check,)
    graph.add_node("judge", judge)
    #graph.add_node("conflict_check", conflict_check)
    graph.add_node("finalize_signal", finalize_signal)

    graph.add_edge(
    START,
    "evidence_similarity_check",
)
    graph.add_conditional_edges("evidence_similarity_check",route_after_similarity,
    {
        "judge": "judge",
        "finalize_signal": "finalize_signal",
    },
    )

    graph.add_edge("judge", "finalize_signal")
    graph.add_edge("finalize_signal", END)

    return graph

# 컴파일된 Assessment 그래프를 생성하고 프로세스에서 재사용한다.
@lru_cache
def get_compiled_graph():

    return build_assessment_graph().compile()

# 충돌 검사 LLM에 전달할 질문·답변·이력·Evidence 프롬프트를 생성한다.
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

"""
