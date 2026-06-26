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
    raise NotImplementedError
