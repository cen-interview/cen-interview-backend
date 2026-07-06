"""Strategy 의 진행 상태.

면접이 진행되는 동안 "지금까지 어떤 주제를, 어떤 난이도로 물었는지"를 유지해
다음 질문이 한 주제에 몰리지 않게 하고 난이도 균형을 맞춘다.
"""

from pydantic import BaseModel, Field

from interview.schemas.question import Difficulty
from interview.schemas.signals import AnswerQuality


class StrategyState(BaseModel):
    asked_topics: list[str] = Field(default_factory=list)
    asked_difficulties: list[Difficulty] = Field(default_factory=list)
    question_count: int = 0


    # 지금까지 실제로 물어본 질문 문장. 
    # 중복 질문 방지용 — LLM에게 "이미 한 질문"으로 전달
    asked_question_texts: list[str] = Field(default_factory=list)
    # 주제별 마지막 답변 평가 결과 (난이도 재조정 시 참고)
    topic_last_quality: dict[str, AnswerQuality] = Field(default_factory=dict)

    def topic_counts(self) -> dict[str, int]:
        """각 토픽 마다 나온 횟수 체크"""
        counts: dict[str, int] = {}
        for t in self.asked_topics:
            counts[t] = counts.get(t, 0) + 1
        return counts
    
    def difficulty_counts(self) -> dict[Difficulty, int]:
        """난이도별 출제 횟수 (분포 균형 판단용)."""
        counts: dict[Difficulty, int] = {}
        for d in self.asked_difficulties:
            counts[d] = counts.get(d, 0) + 1
        return counts

    def recent_topics(self, n: int) -> list[str]:
        """최근 n개 질문의 주제 목록 (연속 주제 회피용)."""
        return self.asked_topics[-n:]
