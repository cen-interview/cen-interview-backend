"""평가 결과 계약.

답변별 평가(AnswerEvaluation)를 누적해 역량 모델(CompetencyModel)을 갱신하고,
면접 종료 후 최종 평가서(FinalReport)를 만든다.
"""

from pydantic import BaseModel, Field

from interview.schemas.signals import AnswerQualitySignal


class AnswerEvaluation(BaseModel):
    """답변 1건에 대한 평가 기록. 매 답변마다 만들어 누적한다."""

    question_id: str
    topic: str
    answer_text: str
    signal: AnswerQualitySignal
    score: float = Field(ge=0.0, le=1.0)  # 정확성/충분성 종합 점수
    notes: str = ""


class CompetencyModel(BaseModel):
    """면접 동안 누적되는 사용자 역량 모델 (강점/약점)."""

    # topic -> 누적 점수 리스트 (평균으로 강/약 판단)
    topic_scores: dict[str, list[float]] = Field(default_factory=dict)

    def record(self, topic: str, score: float) -> None:
        self.topic_scores.setdefault(topic, []).append(score)

    def strengths(self, threshold: float = 0.7) -> list[str]:
        return [t for t, s in self.topic_scores.items()
                if s and (sum(s) / len(s)) >= threshold]

    def weaknesses(self, threshold: float = 0.5) -> list[str]:
        return [t for t, s in self.topic_scores.items()
                if s and (sum(s) / len(s)) < threshold]


class FinalReport(BaseModel):
    """최종 평가서. 화면에 뜨고 결과 저장에도 쓰인다."""

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    topics_to_review: list[str] = Field(default_factory=list)   # 보완 필요 주제
    next_learning: list[str] = Field(default_factory=list)      # 다음 학습 추천
    per_answer: list[AnswerEvaluation] = Field(default_factory=list)
    summary: str = ""
