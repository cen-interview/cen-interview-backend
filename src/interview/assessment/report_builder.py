"""누적된 문항 평가를 바탕으로 면접 최종 리포트를 생성한다.

문항별 AnswerEvaluation의 평균으로 종합 점수를 계산하고,
CompetencyModel과 문항별 평가 내용을 LLM에 전달하여 전체 요약,
강점, 보완 포인트와 추천 학습 방향을 생성한다.

처리 흐름:
    1. 문항별 평가 점수의 평균으로 overall_score를 계산한다.
    2. 오개념 발생 주제와 낮은 점수 주제를 개선 우선순위로 선정한다.
    3. 전체 문항 평가와 역량 상태를 LLM 프롬프트로 변환한다.
    4. LLM의 구조화된 출력을 ReportContent로 받는다.
    5. LLM 호출이 실패하면 규칙 기반 임시 리포트를 반환한다.
    6. 생성된 본문과 문항별 평가를 FinalReport로 조립한다.

최종 리포트 생성이 실패하더라도 면접 종료가 중단되지 않도록
규칙 기반 리포트를 폴백으로 제공한다.
"""

import logging
import re
from time import perf_counter

from pydantic import BaseModel, Field

from interview.schemas.report import (
    AnswerEvaluation,
    CodeAnalysis,
    CompetencyModel,
    FinalReport,
    ReportGenerationResult,
)
from interview.schemas.rubric import RubricCandidate, RubricSource
from interview.schemas.evidence import EvidenceChunk
from interview.assessment.prompts import REPORT_SYSTEM_PROMPT
from interview.llm.client import get_llm
from interview.llm.logging import log_llm_error, log_llm_output


EVIDENCE_MAX_CHUNKS = 3
EVIDENCE_MAX_CHARS_PER_CHUNK = 1500
EVIDENCE_MAX_TOTAL_CHARS = 4000
REPORT_EVIDENCE_MAX_TOTAL_CHARS = 12000

_logger = logging.getLogger("uvicorn.error")


class ReportContent(BaseModel):
    """LLM 또는 임시 로직이 생성하는 최종 리포트 본문 내용.

    Attributes:
        summary:
            면접 전체 요약.

        strengths:
            전체 면접에서 드러난 강점 목록.

        improvement_points:
            전체 면접에서 보완이 필요한 포인트 목록.

        learning_recommendations:
            다음 학습 방향 또는 추천 학습 방법 목록.
    """

    summary: str

    strengths: list[str] = Field(default_factory=list)
    improvement_points: list[str] = Field(default_factory=list)
    learning_recommendations: list[str] = Field(default_factory=list)
    
    evaluation_summaries: list[str] = Field(default_factory=list)
    code_analysis: list[list[CodeAnalysis]] = Field(default_factory=list)
    rubric_candidates: list[RubricCandidate] = Field(default_factory=list)

# 누적 역량과 문항별 평가를 이용해 최종 면접 리포트를 생성한다.
def build_report(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
    evidence_chunks: dict[str, EvidenceChunk] | None = None,
) -> FinalReport:
    """Build a final report without generating shareable rubric rows."""
    return build_report_result(
        competency,
        evaluations,
        evidence_chunks=evidence_chunks,
    ).report


def build_report_result(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
    evidence_chunks: dict[str, EvidenceChunk] | None = None,
    rubric_sources: list[RubricSource] | None = None,
) -> ReportGenerationResult:
    """Build the report and optional rubric rows in one LLM call."""
    evidence_chunks = evidence_chunks or {}
    rubric_sources = rubric_sources or []
    overall_score = _calculate_overall_score(evaluations)

    report_content = _build_content_with_llm(
        competency=competency,
        evaluations=evaluations,
        overall_score=overall_score,
        evidence_chunks=evidence_chunks,
        rubric_sources=rubric_sources,
    )
    
    summarized_evaluations = []

    for index, evaluation in enumerate(evaluations):
        if index < len(report_content.evaluation_summaries):
            summary = report_content.evaluation_summaries[index]
        else:
            summary = evaluation.answer_summary

        if index < len(report_content.code_analysis):
            analyses = report_content.code_analysis[index]
        else:
            analyses = []

        summarized_evaluations.append(
            evaluation.model_copy(
                update={
                    "answer_summary": summary,
                    "code_analysis": analyses,
                }
            )
        )

    report = FinalReport(
        summary=report_content.summary,
        overall_score=overall_score,
        strengths=report_content.strengths,
        improvement_points=report_content.improvement_points,
        learning_recommendations=report_content.learning_recommendations,
        evaluations=summarized_evaluations,
    )
    return ReportGenerationResult(
        report=report,
        rubric_candidates=report_content.rubric_candidates,
    )

# 문항별 평가 점수의 평균을 계산하고 반올림한다.
def _calculate_overall_score(
    evaluations: list[AnswerEvaluation],
) -> float:


    if not evaluations:
        return 0.0

    score_sum = sum(
        evaluation.score
        for evaluation in evaluations
    )

    return round(score_sum / len(evaluations), 0)

# LLM으로 리포트 본문을 생성하고 실패하면 폴백 내용을 반환한다.
def _build_content_with_llm(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
    overall_score: float,
    evidence_chunks: dict[str, EvidenceChunk],
    rubric_sources: list[RubricSource] | None = None,
) -> ReportContent:


    if not evaluations:
        return _temporary_report_content(evaluations)

    try:
        llm = get_llm(temperature=0.2)
        structured_llm = llm.with_structured_output(ReportContent)
        user_prompt = _build_report_user_prompt(
            competency=competency,
            evaluations=evaluations,
            overall_score=overall_score,
            evidence_chunks=evidence_chunks,
            rubric_sources=rubric_sources,
        )

        started_at = perf_counter()
        _logger.info(
            "[LLM][FINAL_REPORT_GENERATION][REQUEST] "
            "evaluation_count=%s rubric_source_count=%s prompt_chars=%s",
            len(evaluations),
            len(rubric_sources or []),
            len(user_prompt),
        )
        result = structured_llm.invoke(
            [
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        result = _remove_external_compatibility_output(result)
        result = _filter_rubric_candidates(
            result,
            rubric_sources=rubric_sources or [],
        )
        log_llm_output(
            "FINAL_REPORT_GENERATION",
            result,
            metadata={
                "overall_score": overall_score,
                "evaluation_count": len(evaluations),
                "elapsed_ms": round(
                    (perf_counter() - started_at) * 1000,
                    2,
                ),
                "prompt_chars": len(user_prompt),
            },
            input_data={"user_prompt": user_prompt},
        )
        return result
    except Exception as exc:
        fallback = _temporary_report_content(evaluations)
        log_llm_error(
            "FINAL_REPORT_GENERATION",
            exc,
            metadata={
                "overall_score": overall_score,
                "evaluation_count": len(evaluations),
            },
            fallback=fallback,
            input_data={
                "competency": competency,
                "evaluations": evaluations,
            },
        )
        return fallback

# 오개념 발생 주제와 낮은 점수 주제를 개선 우선순위로 선정한다.
def _select_topics_to_improve(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
) -> list[str]:
    misconception_topics = []
    low_score_topics = []

    for evaluation in evaluations:
        if any(
            trace.quality == "misconception"
            for trace in evaluation.quality_trace
        ):
            misconception_topics.append(evaluation.topic)

    for topic, score in sorted(
        competency.topic_scores.items(),
        key=lambda item: item[1],
    ):
        low_score_topics.append(topic)

    return _collect_unique_items(
        misconception_topics + low_score_topics
    )

# 역량 상태와 문항별 평가를 최종 리포트 생성 프롬프트로 변환한다.
def _build_report_user_prompt(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
    overall_score: float,
    evidence_chunks: dict[str, EvidenceChunk] | None = None,
    rubric_sources: list[RubricSource] | None = None,
) -> str:
    evidence_chunks = evidence_chunks or {}
    rubric_sources = rubric_sources or []

    topics_to_improve = _select_topics_to_improve(
        competency=competency,
        evaluations=evaluations,
    )

    evaluation_lines = []
    remaining_evidence_chars = REPORT_EVIDENCE_MAX_TOTAL_CHARS

    for index, evaluation in enumerate(evaluations, start=1):
        final_quality_trace = (
            evaluation.quality_trace[-1].model_dump(mode="json")
            if evaluation.quality_trace
            else None
        )
        category = (
            evaluation.question_category.value
            if evaluation.question_category is not None
            else "unknown"
        )
        evidence_context = _build_evidence_context(
            evaluation,
            evidence_chunks,
        )
        if remaining_evidence_chars <= 0:
            evidence_context = "(리포트 Evidence 전체 예산 소진)"
        elif len(evidence_context) > remaining_evidence_chars:
            evidence_context = (
                evidence_context[:remaining_evidence_chars]
                + "\n... [리포트 Evidence 예산 초과로 일부 생략]"
            )
        remaining_evidence_chars = max(
            remaining_evidence_chars - len(evidence_context),
            0,
        )

        evaluation_lines.append(
            (
                f"[문항 {index}]\n"
                f"question_id: {evaluation.question_id}\n"
                f"topic: {evaluation.topic}\n"
                f"question_category: {category}\n"
                f"question: {evaluation.question}\n"
                f"question_evidence_ids: {evaluation.question_evidence_ids}\n"
                f"assessment_evidence_ids: {evaluation.assessment_evidence_ids}\n"
                f"answer_summary: {evaluation.answer_summary}\n"
                f"score: {evaluation.score}\n"
                f"comment: {evaluation.comment}\n"
                f"delivery_note: {evaluation.delivery_note or '(없음)'}\n"
                f"final_quality_trace: {final_quality_trace}\n"
                f"evidence: {evidence_context}"
            )
        )

    return (
        f"overall_score: {overall_score}\n"
        f"topics_to_improve: {topics_to_improve}\n"
        f"competency.average_score: {competency.average_score}\n"
        f"competency.topic_scores: {competency.topic_scores}\n"
        f"competency.strengths: {competency.strengths}\n"
        f"competency.improvement_points: {competency.improvement_points}\n"
        f"competency.learning_recommendations: {competency.learning_recommendations}\n\n"
        "[문항별 evaluations]\n"
        + "\n\n".join(evaluation_lines)
        + f"\n\n{_build_rubric_generation_context(rubric_sources)}"
    )


def _build_rubric_generation_context(
    rubric_sources: list[RubricSource],
) -> str:
    """Build an allow-listed rubric request for the report LLM."""
    if not rubric_sources:
        return (
            "[Rubric 생성 요청]\n"
            "공유 동의가 없거나 새 질문이 없습니다. "
            "rubric_candidates는 반드시 빈 배열로 반환하세요."
        )

    rendered_sources = "\n\n".join(
        (
            f"question_id: {source.question_id}\n"
            f"topic: {source.topic}\n"
            f"question: {source.question}\n"
            f"answer: {source.answer}"
        )
        for source in rubric_sources
    )
    return (
        "[Rubric 생성 요청]\n"
        "사용자가 공유에 동의했습니다. 아래 질문에 대해서만 재사용 가능한 "
        "평가 기준을 생성하세요. 질문당 3~5개 기준을 작성하고, 핵심 정답 "
        "요소는 required=true, 예시와 부가 설명은 required=false로 두세요. "
        "사용자 이름, 프로젝트명, 개인정보는 제외하고 question_id를 변경하지 "
        "마세요.\n\n"
        f"{rendered_sources}"
    )


def _filter_rubric_candidates(
    report_content: ReportContent,
    *,
    rubric_sources: list[RubricSource],
) -> ReportContent:
    """Drop hallucinated rows and restore canonical source metadata."""
    source_by_id = {
        source.question_id: source
        for source in rubric_sources
    }
    candidates: list[RubricCandidate] = []
    seen_ids: set[str] = set()
    for candidate in report_content.rubric_candidates:
        source = source_by_id.get(candidate.question_id)
        if (
            source is None
            or candidate.question_id in seen_ids
            or not candidate.criteria
        ):
            continue
        seen_ids.add(candidate.question_id)
        candidates.append(candidate.model_copy(update={
            "topic": source.topic,
            "question": source.question,
        }))
    return report_content.model_copy(
        update={"rubric_candidates": candidates}
    )


def _matched_evidence_chunks(
    evaluation: AnswerEvaluation,
    evidence_chunks: dict[str, EvidenceChunk],
) -> list[EvidenceChunk]:
    """Return the concrete Evidence chunks linked to one evaluation."""
    evidence_ids = list(dict.fromkeys(
        evaluation.assessment_evidence_ids
        + evaluation.question_evidence_ids
    ))
    return [
        evidence_chunks[evidence_id]
        for evidence_id in evidence_ids
        if evidence_id in evidence_chunks
    ]


def _remove_external_compatibility_output(
    report_content: ReportContent,
) -> ReportContent:
    """Keep the response contract while disabling external-doc fields."""
    normalized_rows = [
        [
            analysis.model_copy(
                update={
                    "current_code": _bounded_code_excerpt(
                        analysis.current_code
                    ),
                    "compatibility_status": "not_evaluated",
                    "modern_code": None,
                    "references": [],
                }
            )
            for analysis in analyses
        ]
        for analyses in report_content.code_analysis
    ]
    return report_content.model_copy(
        update={"code_analysis": normalized_rows}
    )


def _build_evidence_context(
    evaluation: AnswerEvaluation,
    evidence_chunks: dict[str, EvidenceChunk],
) -> str:
    """현재 리포트 생성 과정에서 확보된 Evidence 원문을 문항별로 구성한다."""
    matched_chunks = _matched_evidence_chunks(
        evaluation,
        evidence_chunks,
    )[:EVIDENCE_MAX_CHUNKS]

    if not matched_chunks:
        return "(현재 실행에서 확보된 Evidence 원문 없음)"

    rendered_chunks: list[str] = []
    remaining_chars = EVIDENCE_MAX_TOTAL_CHARS
    for chunk in matched_chunks:
        content = _redact_secrets(chunk.text)
        content_truncated = len(content) > EVIDENCE_MAX_CHARS_PER_CHUNK
        content = content[:EVIDENCE_MAX_CHARS_PER_CHUNK]
        if content_truncated:
            content += "\n... [코드 일부 생략]"

        rendered = (
            f"- chunk_id: {chunk.chunk_id}\n"
            f"  source_type: {chunk.source_type.value}\n"
            f"  source_file: {chunk.file_path or '(없음)'}\n"
            f"  language: {chunk.language or '(없음)'}\n"
            f"  content:\n{content}"
        )
        if len(rendered) > remaining_chars:
            rendered = rendered[:remaining_chars] + "\n... [Evidence 일부 생략]"
        rendered_chunks.append(rendered)
        remaining_chars -= len(rendered)
        if remaining_chars <= 0:
            break

    return "\n\n".join(rendered_chunks)


_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b[\w.-]*(?:api[_-]?key|secret|token|password|authorization)"
    r"[\w.-]*\b\s*[:=]\s*)(['\"`])([^'\"`\r\n]+)(['\"`])"
)


def _redact_secrets(text: str) -> str:
    """Mask credentials before project code is sent to an LLM or logged."""
    return _SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}"
            f"[REDACTED]{match.group(4)}"
        ),
        text,
    )


def _bounded_code_excerpt(text: str) -> str:
    """Redact and cap code copied into the final API response."""
    redacted = _redact_secrets(text)
    if len(redacted) <= EVIDENCE_MAX_CHARS_PER_CHUNK:
        return redacted
    return (
        redacted[:EVIDENCE_MAX_CHARS_PER_CHUNK]
        + "\n... [코드 일부 생략]"
    )

# LLM 호출 실패 시 사용할 규칙 기반 리포트 본문을 생성한다.
def _temporary_report_content(
    evaluations: list[AnswerEvaluation],
) -> ReportContent:


    if not evaluations:
        return ReportContent(
            summary="평가할 답변 기록이 없습니다.",
            strengths=[],
            improvement_points=["답변 기록이 없어 보완 포인트를 산정할 수 없습니다."],
            learning_recommendations=[
                "질문에 답변한 후 다시 평가를 진행해 주세요.",
            ],
        )
    low_score_topics = _collect_unique_items(
        evaluation.topic
        for evaluation in evaluations
        if evaluation.score < 70
    )

    return ReportContent(
        summary=(
            f"총 {len(evaluations)}개의 문항을 평가했습니다. "
            f"평균 점수는 {_calculate_overall_score(evaluations)}점이며, "
            "문항별 답변 요약과 평가 코멘트를 바탕으로 전체 리포트를 생성했습니다."
        ),
        strengths=[
            "면접 질문에 대해 답변을 이어가며 파생 질문을 통해 내용을 보완했습니다."
        ],
        improvement_points=(
            [
                f"{topic} 주제의 설명 정확도와 구체성을 보완할 필요가 있습니다."
                for topic in low_score_topics
            ]
            or [
                "핵심 개념을 정의, 사용 이유, 실제 적용 사례 순서로 더 구조화해 설명하면 좋습니다."
            ]
        ),
        learning_recommendations=[
            "핵심 개념을 정의, 사용 이유, 실제 적용 사례, 한계점 순서로 정리해 보세요.",
            "프로젝트 경험을 설명할 때 기술 선택 이유와 트러블슈팅 과정을 함께 말해 보세요.",
        ],
    )

# 입력 순서를 유지하면서 중복 문자열을 제거한다.
def _collect_unique_items(
    items,
) -> list[str]:


    return list(dict.fromkeys(items))
