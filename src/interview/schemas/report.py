"""
AnswerEvaluation / CompetencyModel / FinalReport

Assessment 가 매 답변마다 AnswerEvaluation 을 만들고, 이를 CompetencyModel 에
누적해 강점/약점을 갱신한다. 면접 종료 시 finalize() 가 FinalReport 를 만든다.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# signals.py 의 quality 값을 재사용한다 (같은 4종을 일관되게 쓰기 위함)
from .signals import AnswerQuality


class AnswerEvaluation(BaseModel):
    """한 답변에 대한 평가 1건."""
    question_id: str
    topic: str
    quality: AnswerQuality
    accuracy: float = Field(ge=0.0, le=1.0)     # 정확성
    sufficiency: float = Field(ge=0.0, le=1.0)  # 설명 충분성
    key_concepts: list[str] = Field(default_factory=list)  # 답변이 짚은 핵심 개념
    comment: str | None = None

    # (음성 전용) 전달력 보조 평가. 채팅이면 None.
    delivery_note: str | None = None


class CompetencyModel(BaseModel):
    """면접 내내 누적되는 역량 상태 (강점/약점)."""
    # 주제별 점수 누적 (예: {"JPA": 0.8, "JWT": 0.4})
    topic_scores: dict[str, float] = Field(default_factory=dict)

    def record(self, topic: str, score: float) -> None:
        """주제별 점수를 누적 평균으로 갱신한다."""
        previous = self.topic_scores.get(topic)
        if previous is None:
            self.topic_scores[topic] = score
            return
        self.topic_scores[topic] = (previous + score) / 2

    def strengths(self, threshold: float = 0.7) -> list[str]:
        """강점으로 볼 수 있는 주제 목록."""
        return [
            topic
            for topic, score in self.topic_scores.items()
            if score >= threshold
        ]

    def weaknesses(self, threshold: float = 0.5) -> list[str]:
        """보완이 필요한 주제 목록."""
        return [
            topic
            for topic, score in self.topic_scores.items()
            if score < threshold
        ]


class FinalReport(BaseModel):
    """종료 후 사용자에게 보여줄 최종 평가서."""
    strengths: list[str]
    weaknesses: list[str]
    topics_to_improve: list[str]          # 보완이 필요한 기술 주제
    learning_recommendations: list[str]   # 다음 학습 추천
    evaluations: list[AnswerEvaluation] = Field(default_factory=list)  # 문항별 평가 모음
