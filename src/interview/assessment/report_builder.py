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
      - 약점 기반 next_learning 추천 (LLM 으로 문장화 가능)
      - summary 한 단락 생성
    """
    # [현재 Stub 작동] competency 에 이미 누적된 strengths/weaknesses 를 그대로 사용
    weaknesses = competency.weaknesses or ["[Stub] 데이터 부족으로 약점 미산출"]
    return FinalReport(
        strengths=competency.strengths or ["[Stub] 데이터 부족으로 강점 미산출"],
        weaknesses=weaknesses,
        topics_to_improve=weaknesses,
        learning_recommendations=[f"[Stub] '{w}' 관련 자료를 복습해보세요." for w in weaknesses],
        evaluations=evaluations,
    )
