"""최종 평가서 생성.

누적된 역량 모델(CompetencyModel)을 바탕으로 강점/약점/보완 주제/학습 추천을
정리해 FinalReport 를 만든다.
"""

from interview.schemas.report import AnswerEvaluation, CompetencyModel, FinalReport


def build_report(
    competency: CompetencyModel, evaluations: list[AnswerEvaluation]
) -> FinalReport:
    """최종 리포트 작성.

    TODO(담당 D):
      - competency.strengths()/weaknesses() 로 강·약점 정리
      - 약점 주제 → topics_to_review
      - 약점 기반 next_learning 추천 (LLM 으로 문장화 가능)
      - summary 한 단락 생성
    """
    strengths = competency.strengths()
    weaknesses = competency.weaknesses()

    if not strengths:
        strengths = ["기본 답변 흐름"]
    if not weaknesses:
        weaknesses = ["구체적인 근거 설명"]

    recommendations = [
        f"{topic} 관련 핵심 개념을 정리하고 프로젝트 경험과 연결해 설명해보세요."
        for topic in weaknesses
    ]

    return FinalReport(
        strengths=strengths,
        weaknesses=weaknesses,
        topics_to_improve=weaknesses,
        learning_recommendations=recommendations,
        evaluations=evaluations,
    )
