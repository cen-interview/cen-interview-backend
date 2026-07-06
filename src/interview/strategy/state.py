"""Strategy 의 진행 상태.

면접이 진행되는 동안 "지금까지 어떤 주제를, 어떤 난이도로 물었는지"를 유지해
다음 질문이 한 주제에 몰리지 않게 하고 난이도 균형을 맞춘다.
"""
# 최대 꼬리질문 3개

from pydantic import BaseModel, Field

from interview.schemas.question import Difficulty


class StrategyState(BaseModel):
    asked_topics: list[str] = Field(default_factory=list)
    asked_difficulties: list[Difficulty] = Field(default_factory=list)
    question_count: int = 0

    def topic_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.asked_topics:
            counts[t] = counts.get(t, 0) + 1
        return counts
