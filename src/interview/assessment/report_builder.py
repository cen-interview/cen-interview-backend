"""최종 평가서 생성.

누적된 역량 모델과 문항별 평가 결과를 바탕으로
면접 전체 요약, 종합 점수, 전체 강점, 전체 보완 포인트,
추천 학습 방향을 생성한다.

현재는 LLM 연결 전이므로 임시 리포트 내용을 반환한다.
"""

from pydantic import BaseModel, Field

from interview.schemas.report import (
    AnswerEvaluation,
    CompetencyModel,
    FinalReport,
)
from interview.assessment.prompts import REPORT_SYSTEM_PROMPT
from interview.llm.client import get_llm

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


def build_report(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
) -> FinalReport:
    """누적 평가를 바탕으로 최종 리포트를 생성한다.

    Args:
        competency:
            면접 전체의 누적 역량 상태.
            주제별 점수, 강점, 보완 포인트 등을 담는다.

        evaluations:
            문항별 평가 목록.
            각 AnswerEvaluation은 메인 질문, 전체 답변 요약,
            문항 점수, 평가 코멘트를 포함한다.

    Returns:
        FinalReport:
            사용자에게 최종적으로 보여줄 면접 평가 리포트.
    """

    overall_score = _calculate_overall_score(evaluations)

    report_content = _build_content_with_llm(
        competency=competency,
        evaluations=evaluations,
        overall_score=overall_score,
    )

    return FinalReport(
        summary=report_content.summary,
        overall_score=overall_score,
        strengths=report_content.strengths,
        improvement_points=report_content.improvement_points,
        learning_recommendations=report_content.learning_recommendations,
        evaluations=evaluations,
    )


def _calculate_overall_score(
    evaluations: list[AnswerEvaluation],
) -> float:
    """문항별 점수의 평균을 계산한다.

    Args:
        evaluations:
            문항별 평가 목록.

    Returns:
        float:
            전체 문항 점수의 평균.
            평가가 없으면 0.0을 반환한다.

    Note:
        round(..., 2)의 2는 문항 개수가 아니라
        소수점 둘째 자리까지 반올림한다는 의미이다.
    """

    if not evaluations:
        return 0.0

    score_sum = sum(
        evaluation.score
        for evaluation in evaluations
    )

    return round(score_sum / len(evaluations), 0)


def _build_content_with_llm(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
    overall_score: float,
) -> ReportContent:
    """LLM으로 최종 리포트 본문을 생성한다.

    Args:
        competency:
            면접 전체의 누적 역량 상태.

        evaluations:
            문항별 평가 목록.

        overall_score:
            문항별 점수 평균으로 계산된 종합 점수.

    Returns:
        ReportContent:
            FinalReport에 들어갈 summary, strengths,
            improvement_points, learning_recommendations.

    TODO:
        현재는 LLM 연결 전이므로 임시 생성 함수를 호출한다.
        추후 이 함수 내부만 실제 LLM 호출로 교체하면 된다.
    """

    if not evaluations:
        return _temporary_report_content(evaluations)

    try:
        llm = get_llm(temperature=0.2)
        structured_llm = llm.with_structured_output(ReportContent)

        return structured_llm.invoke(
        [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_report_user_prompt(
                    competency=competency,
                    evaluations=evaluations,
                    overall_score=overall_score,
                ),
            },
        ]
    )
    except Exception:
        return _temporary_report_content(evaluations)

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
    

def _build_report_user_prompt(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
    overall_score: float,
) -> str:
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

        evaluation_lines.append(
            (
                f"[문항 {index}]\n"
                f"question_id: {evaluation.question_id}\n"
                f"topic: {evaluation.topic}\n"
                f"question: {evaluation.question}\n"
                f"answer_summary: {evaluation.answer_summary}\n"
                f"score: {evaluation.score}\n"
                f"comment: {evaluation.comment}\n"
                f"delivery_note: {evaluation.delivery_note or '(없음)'}\n"
                f"quality_trace: {quality_trace}"
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
    
    
def _temporary_report_content(
    evaluations: list[AnswerEvaluation],
) -> ReportContent:
    """LLM 연결 전 사용하는 임시 리포트 내용을 생성한다.

    Args:
        evaluations:
            문항별 평가 목록.

    Returns:
        ReportContent:
            임시 면접 요약, 강점, 보완 포인트, 학습 추천.
    """

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


def _collect_unique_items(
    items,
) -> list[str]:
    """중복을 제거하면서 기존 순서를 유지한다.

    Args:
        items:
            중복 제거 대상 iterable.

    Returns:
        list[str]:
            입력 순서를 유지한 고유 문자열 목록.
    """

    return list(dict.fromkeys(items))
