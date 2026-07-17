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

from pydantic import BaseModel, Field

from interview.schemas.report import (
    AnswerEvaluation,
    CodeAnalysis,
    CompetencyModel,
    FinalReport,
)
from interview.assessment.prompts import REPORT_SYSTEM_PROMPT
from interview.llm.client import get_llm
from interview.llm.logging import log_llm_error, log_llm_output
from interview.schemas.evidence import EvidenceChunk
from interview.schemas.question import QuestionCategory

EVIDENCE_MAX_CHUNKS = 3
EVIDENCE_MAX_CHARS_PER_CHUNK = 2000

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
    # evaluations와 같은 순서의 문항별 코드 분석
    # 바깥 list는 문항, 안쪽 list는 해당 문항의 코드 분석 목록
    code_analysis: list[list[CodeAnalysis]] = Field(
        default_factory=list
    )

# 누적 역량과 문항별 평가를 이용해 최종 면접 리포트를 생성한다.
def build_report(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
    evidence_chunks: dict[str, EvidenceChunk] | None = None,
) -> FinalReport:
    evidence_chunks = evidence_chunks or {}


    overall_score = _calculate_overall_score(evaluations)

    report_content = _build_content_with_llm(
        competency=competency,
        evaluations=evaluations,
        overall_score=overall_score,
        evidence_chunks = evidence_chunks,
    )
    
    summarized_evaluations: list[AnswerEvaluation] = []

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

    return FinalReport(
        summary=report_content.summary,
        overall_score=overall_score,
        strengths=report_content.strengths,
        improvement_points=report_content.improvement_points,
        learning_recommendations=report_content.learning_recommendations,
        evaluations=summarized_evaluations,
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
def _normalize_code_analysis_output(
    content: ReportContent,
    evaluations: list[AnswerEvaluation],
    evidence_chunks: dict[str, EvidenceChunk],
) -> ReportContent:
    """Evidence가 있는 PROJECT 문항의 코드 분석만 유지한다."""

    normalized_rows: list[list[CodeAnalysis]] = []

    for index, evaluation in enumerate(evaluations):
        has_project_evidence = (
            evaluation.question_category
            == QuestionCategory.PROJECT
            and bool(
                _matched_evidence_chunks(
                    evaluation,
                    evidence_chunks,
                )
            )
        )

        if not has_project_evidence:
            normalized_rows.append([])
            continue

        analyses = (
            content.code_analysis[index]
            if index < len(content.code_analysis)
            else []
        )

        normalized_rows.append([
            analysis.model_copy(
                update={
                    "current_code": analysis.current_code[
                        :EVIDENCE_MAX_CHARS_PER_CHUNK
                    ],
                    "compatibility_status": "not_evaluated",
                    "references": [],
                }
            )
            for analysis in analyses
        ])

    return content.model_copy(
        update={
            "code_analysis": normalized_rows,
        }
    )
# LLM으로 리포트 본문을 생성하고 실패하면 폴백 내용을 반환한다.
def _build_content_with_llm(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
    overall_score: float,
    evidence_chunks: dict[str, EvidenceChunk],
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
        )

        result = structured_llm.invoke(
            [
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        result = _normalize_code_analysis_output(
            content=result,
            evaluations=evaluations,
            evidence_chunks=evidence_chunks,
            )
        log_llm_output(
            "FINAL_REPORT_GENERATION",
            result,
            metadata={
                "overall_score": overall_score,
                "evaluation_count": len(evaluations),
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

def _matched_evidence_chunks(
    evaluation: AnswerEvaluation,
    evidence_chunks: dict[str, EvidenceChunk],
) -> list[EvidenceChunk]:
    """문항 생성 또는 답변 평가에 실제 사용된 Evidence를 반환한다."""

    evidence_ids = list(
        dict.fromkeys(
            evaluation.assessment_evidence_ids
            + evaluation.question_evidence_ids
        )
    )

    return [
        evidence_chunks[evidence_id]
        for evidence_id in evidence_ids
        if evidence_id in evidence_chunks
    ]
    
def _build_evidence_context(
    evaluation: AnswerEvaluation,
    evidence_chunks: dict[str, EvidenceChunk],
) -> str:
    """PROJECT 문항에 연결된 Evidence 원문을 프롬프트로 만든다."""

    if evaluation.question_category != QuestionCategory.PROJECT:
        return "(기술 질문 — 코드 Evidence 분석 대상 아님)"

    matched_chunks = _matched_evidence_chunks(
        evaluation,
        evidence_chunks,
    )[:EVIDENCE_MAX_CHUNKS]

    if not matched_chunks:
        return "(PROJECT 질문이지만 확보된 Evidence 원문 없음)"

    rendered_chunks: list[str] = []

    for chunk in matched_chunks:
        code = chunk.text[:EVIDENCE_MAX_CHARS_PER_CHUNK]

        rendered_chunks.append(
            (
                f"- chunk_id: {chunk.chunk_id}\n"
                f"  source_file: {chunk.file_path or '(없음)'}\n"
                f"  language: {chunk.language or '(없음)'}\n"
                f"  topic: {chunk.topic}\n"
                f"  code:\n{code}"
            )
        )

    return "\n".join(rendered_chunks)

# 역량 상태와 문항별 평가를 최종 리포트 생성 프롬프트로 변환한다.
def _build_report_user_prompt(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
    overall_score: float,
    evidence_chunks: dict[str, EvidenceChunk] | None = None,
) -> str:
    evidence_chunks = evidence_chunks or {}
    topics_to_improve = _select_topics_to_improve(
        competency=competency,
        evaluations=evaluations,
    )

    evaluation_lines = []

    for index, evaluation in enumerate(evaluations, start=1):
        quality_trace = [
            trace.model_dump(mode="json")
            for trace in evaluation.quality_trace
        ]
        category = (
            evaluation.question_category.value
            if evaluation.question_category is not None
            else "unknown"
            )

        evidence_context = _build_evidence_context(
            evaluation,
            evidence_chunks,
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
                f"quality_trace: {quality_trace}\n"
                f"evidence:\n{evidence_context}"
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
