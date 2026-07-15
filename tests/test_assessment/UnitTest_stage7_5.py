"""Assessment 7-5 단위 테스트.

최종 리포트 LLM 호출이 실패하면 기존 스텁 리포트로 폴백하는지 확인한다.
"""

from interview.assessment import report_builder
from interview.schemas.report import (
    AnswerEvaluation,
    CompetencyModel,
    QualityTrace,
)


def make_evaluation(
    *,
    question_id: str = "q-jpa-1",
    topic: str = "JPA",
    score: float = 62.0,
) -> AnswerEvaluation:
    return AnswerEvaluation(
        question_id=question_id,
        topic=topic,
        question="JPA N+1 문제가 왜 발생하고 어떻게 해결할 수 있나요?",
        answer_summary=(
            "N+1 원인과 해결책을 설명했지만 일부 설명이 부족했습니다."
        ),
        score=score,
        comment="오개념 정정 이력이 있는 문항입니다.",
        quality_trace=[
            QualityTrace(
                question_kind="main",
                quality="misconception",
                target="N+1 발생 원인",
                rationale=["초기 답변에서 N+1 발생 원인을 혼동했습니다."],
            ),
            QualityTrace(
                question_kind="challenge",
                quality="sufficient",
                target=None,
                rationale=["압박 질문 이후 핵심 원인을 정정했습니다."],
            ),
        ],
    )


def test_stage7_5_falls_back_to_stub_report_when_llm_fails(
    monkeypatch,
) -> None:
    """7-5. LLM 호출 실패 시 현행 스텁 리포트가 반환된다."""

    def fail_get_llm(*args, **kwargs):
        raise RuntimeError("테스트용 LLM 장애")

    monkeypatch.setattr(
        report_builder,
        "get_llm",
        fail_get_llm,
    )

    evaluation = make_evaluation()
    competency = CompetencyModel(
        topic_scores={
            "JPA": evaluation.score,
        },
        average_score=evaluation.score,
    )

    report = report_builder.build_report(
        competency=competency,
        evaluations=[evaluation],
    )

    print("\n===== 7-5 LLM 실패 후 폴백 리포트 =====")
    print("summary:", report.summary)
    print("overall_score:", report.overall_score)
    print("strengths:", report.strengths)
    print("improvement_points:", report.improvement_points)
    print("learning_recommendations:", report.learning_recommendations)

    assert report.overall_score == 62.0
    assert report.summary == (
        "총 1개의 문항을 평가했습니다. "
        "평균 점수는 62.0점이며, "
        "문항별 답변 요약과 평가 코멘트를 바탕으로 전체 리포트를 생성했습니다."
    )
    assert report.strengths == [
        "면접 질문에 대해 답변을 이어가며 파생 질문을 통해 내용을 보완했습니다."
    ]
    assert report.improvement_points == [
        "JPA 주제의 설명 정확도와 구체성을 보완할 필요가 있습니다."
    ]
    assert report.learning_recommendations == [
        "핵심 개념을 정의, 사용 이유, 실제 적용 사례, 한계점 순서로 정리해 보세요.",
        "프로젝트 경험을 설명할 때 기술 선택 이유와 트러블슈팅 과정을 함께 말해 보세요.",
    ]
    assert report.evaluations == [evaluation]


def test_stage7_5_empty_evaluations_still_returns_empty_stub_report(
    monkeypatch,
) -> None:
    """7-5. 평가가 없을 때도 LLM 없이 기존 빈 리포트 스텁을 반환한다."""

    def fail_get_llm(*args, **kwargs):
        raise AssertionError("evaluations가 없으면 LLM을 호출하지 않아야 합니다.")

    monkeypatch.setattr(
        report_builder,
        "get_llm",
        fail_get_llm,
    )

    report = report_builder.build_report(
        competency=CompetencyModel(),
        evaluations=[],
    )

    print("\n===== 7-5 빈 evaluations 폴백 리포트 =====")
    print("summary:", report.summary)
    print("overall_score:", report.overall_score)
    print("improvement_points:", report.improvement_points)

    assert report.overall_score == 0.0
    assert report.summary == "평가할 답변 기록이 없습니다."
    assert report.strengths == []
    assert report.improvement_points == [
        "답변 기록이 없어 보완 포인트를 산정할 수 없습니다."
    ]
    assert report.learning_recommendations == [
        "질문에 답변한 후 다시 평가를 진행해 주세요."
    ]
    assert report.evaluations == []
