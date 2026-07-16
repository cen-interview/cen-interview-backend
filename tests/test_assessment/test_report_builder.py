from interview.assessment import report_builder
from interview.assessment.report_builder import ReportContent
from interview.schemas.evidence import EvidenceChunk, SourceType
from interview.schemas.question import QuestionCategory
from interview.schemas.report import (
    AnswerEvaluation,
    CodeAnalysis,
    CompetencyModel,
    QualityTrace,
)


def _evaluation(evidence_ids: list[str] | None = None) -> AnswerEvaluation:
    return AnswerEvaluation(
        question_id="q-project",
        topic="프로젝트 코드",
        question="현재 코드를 설명해 주세요.",
        answer_summary="현재 코드의 역할을 설명했습니다.",
        score=80,
        comment="현재 프로젝트 코드를 기준으로 평가했습니다.",
        question_category=QuestionCategory.PROJECT,
        assessment_evidence_ids=evidence_ids or [],
        quality_trace=[
            QualityTrace(question_kind="main", quality="insufficient"),
            QualityTrace(question_kind="follow_up", quality="sufficient"),
        ],
    )


def _chunk(chunk_id: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        source_type=SourceType.GITHUB,
        source_url=f"https://example.com/{chunk_id}.js",
        topic="프로젝트 코드",
        doc_type="코드",
        confidence=0.9,
        file_path=f"src/{chunk_id}.js",
        language="javascript",
    )


def _analysis() -> CodeAnalysis:
    return CodeAnalysis(
        topic="프로젝트 코드",
        source_file="src/example.js",
        current_code="const value = 1",
        code_assessment="현재 코드의 역할을 분석했습니다.",
        answer_status="answered",
        expected_answer="현재 코드의 입력과 출력을 설명해야 합니다.",
        compatibility_status="current_valid",
        modern_code="const value = 2",
        improvement_reason="현재 코드의 개선점을 설명합니다.",
        references=["https://example.com/docs"],
    )


def test_evidence_context_redacts_secrets_and_limits_size():
    secret = "example-sensitive-value"
    chunks = {
        f"chunk-{index}": _chunk(
            f"chunk-{index}",
            (
                f"const KAKAO_REST_API_KEY = '{secret}'\n"
                "const payload = 'x'.repeat(3000)\n"
                + ("x" * 3000)
            ),
        )
        for index in range(4)
    }
    evaluation = _evaluation(list(chunks))

    context = report_builder._build_evidence_context(evaluation, chunks)

    assert secret not in context
    assert "[REDACTED]" in context
    assert "chunk-3" not in context
    assert len(context) <= report_builder.EVIDENCE_MAX_TOTAL_CHARS + 100


def test_report_prompt_uses_only_final_quality_trace_and_current_evidence():
    evaluation = _evaluation()

    prompt = report_builder._build_report_user_prompt(
        competency=CompetencyModel(),
        evaluations=[evaluation],
        overall_score=80,
    )

    assert "final_quality_trace" in prompt
    assert "follow_up" in prompt
    assert "insufficient" not in prompt
    assert "[문항별 evaluations]" in prompt


def test_external_compatibility_fields_are_always_removed():
    secret = "output-sensitive-value"
    analysis_with_long_code = _analysis().model_copy(
        update={
            "current_code": (
                f"const API_KEY = '{secret}'\n" + ("x" * 3000)
            )
        }
    )
    content = ReportContent(
        summary="summary",
        code_analysis=[[analysis_with_long_code]],
    )

    normalized = report_builder._remove_external_compatibility_output(content)
    analysis = normalized.code_analysis[0][0]

    assert analysis.compatibility_status == "not_evaluated"
    assert analysis.modern_code is None
    assert analysis.references == []
    assert secret not in analysis.current_code
    assert len(analysis.current_code) <= (
        report_builder.EVIDENCE_MAX_CHARS_PER_CHUNK + 30
    )
