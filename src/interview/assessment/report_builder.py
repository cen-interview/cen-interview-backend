"""최종 평가서 생성.

누적된 역량 모델(CompetencyModel)을 바탕으로 강점/약점/보완 주제/학습 추천을
정리해 FinalReport를 만든다.
"""

from interview.schemas.report import AnswerEvaluation, CompetencyModel, FinalReport


def build_report(
    competency: CompetencyModel,
    evaluations: list[AnswerEvaluation],
) -> FinalReport:
    """최종 리포트 작성 스텁."""


    if evaluations:
        overall_score = sum(e.score for e in evaluations) / len(evaluations)
    else:
        overall_score = 0.0

    weak_topics = [
        topic
        for topic, score in competency.topic_scores.items()
        if score < 70
    ]

    return FinalReport(
        overall_score=overall_score,
        summary="임시 평가 요약입니다. 문항별 답변을 바탕으로 강점과 보완점을 정리했습니다.",
        strengths=competency.strengths or [
            "질문에 대해 답변을 시도하고 후속 질문에 응답하는 흐름이 확인되었습니다."
        ],
        weaknesses=competency.weaknesses or [
            "일부 답변에서 핵심 개념의 원인, 한계, 실제 적용 방식 설명이 부족할 수 있습니다."
        ],
        topics_to_improve=weak_topics or [
            "FastAPI"
        ],
        learning_recommendations=[
            "핵심 개념을 정의 → 발생 원인 → 실제 적용 사례 → 한계점 순서로 정리해보세요.",
            "프로젝트 경험을 설명할 때 사용 이유와 트러블슈팅 과정을 함께 말하는 연습이 필요합니다.",
        ],
        evaluations=evaluations,
    )
