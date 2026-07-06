"""최종 평가서 생성.

누적된 역량 모델과 질문 세트별 평가를 바탕으로
최종 점수, 강점, 약점, 보완 주제, 학습 추천을 생성한다.

현재는 LLM 연결 전이므로 임시 리포트 내용을 반환한다.
"""

from pydantic import BaseModel, Field

from interview.schemas.report import (
    AnswerEvaluation,
    CompetencyModel,
    FinalReport,
)


class ReportContent(BaseModel):
    """LLM이 생성할 최종 리포트 내용."""

    summary: str

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    learning_recommendations: list[str] = Field(
        default_factory=list
    )


def build_report(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
) -> FinalReport:
    """누적 평가를 바탕으로 최종 리포트를 생성한다."""


    overall_score = _calculate_overall_score(evaluations)


    topics_to_improve = _find_topics_to_improve(
        competency=competency,
        evaluations=evaluations,
    )

    report_content = _build_content_with_llm(
        competency=competency,
        evaluations=evaluations,
        overall_score=overall_score,
        topics_to_improve=topics_to_improve,
    )

    return FinalReport(
        overall_score=overall_score,
        summary=report_content.summary,
        strengths=report_content.strengths,
        weaknesses=report_content.weaknesses,
        topics_to_improve=topics_to_improve,
        learning_recommendations=(
            report_content.learning_recommendations
        ),
        evaluations=evaluations,
    )


def _calculate_overall_score(
    evaluations: list[AnswerEvaluation],
) -> float:
    """질문 세트 점수의 평균을 계산한다."""

    if not evaluations:
        return 0.0

    score_sum = sum(
        evaluation.score
        for evaluation in evaluations
    )

    return round(score_sum / len(evaluations), 2)


def _find_topics_to_improve(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
) -> list[str]:
    """점수가 낮거나 오개념이 발생한 주제를 찾는다."""

    weak_topics = {
        topic
        for topic, score in competency.topic_scores.items()
        if score < 70
    }

    # 오개념 또는 충돌이 남은 주제도 보완 대상으로 포함한다.
    for evaluation in evaluations:
        if evaluation.quality.value in {
            "misconception",
            "confirm_negative",
        }:
            weak_topics.add(evaluation.topic)

    return sorted(weak_topics)


def _build_content_with_llm(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
    overall_score: float,
    topics_to_improve: list[str],
) -> ReportContent:
    """LLM으로 최종 리포트 내용을 생성한다.

    현재는 LLM 연결 전이므로 임시 생성 함수를 호출한다.
    향후 이 함수 내부만 실제 LLM 호출로 교체한다.
    """

    # 실제 LLM 연결 시 프롬프트에 전달할 값
    _ = competency
    _ = overall_score
    _ = topics_to_improve

    return _temporary_report_content(evaluations)


def _temporary_report_content(
    evaluations: list[AnswerEvaluation],
) -> ReportContent:
    """LLM 연결 전 사용하는 임시 리포트 내용."""

    if not evaluations:
        return ReportContent(
            summary="평가할 답변 기록이 없습니다.",
            strengths=[],
            weaknesses=["답변 기록 없음"],
            learning_recommendations=[
                "질문에 답변한 후 다시 평가를 진행해 주세요.",
            ],
        )

    strengths = _collect_unique_items(
        item
        for evaluation in evaluations
        for item in evaluation.strengths
    )

    weaknesses = _collect_unique_items(
        item
        for evaluation in evaluations
        for item in evaluation.improvements
    )

    return ReportContent(
        summary=(
            f"총 {len(evaluations)}개의 질문 세트를 평가했습니다. "
            "문항별 답변을 바탕으로 강점과 보완점을 정리했습니다."
        ),
        strengths=strengths or [
            "질문에 답변하고 파생 질문을 통해 내용을 보완했습니다."
        ],
        weaknesses=weaknesses or [
            "핵심 개념의 실제 적용 사례와 한계점 설명이 필요합니다."
        ],
        learning_recommendations=[
            (
                "핵심 개념을 정의, 사용 이유, 실제 적용 사례, "
                "한계점 순서로 정리해 보세요."
            ),
            (
                "프로젝트 경험을 설명할 때 선택 이유와 "
                "트러블슈팅 과정을 함께 말해 보세요."
            ),
        ],

    )


def _collect_unique_items(
    items,
) -> list[str]:
    """중복을 제거하면서 기존 순서를 유지한다."""

    return list(dict.fromkeys(items))

