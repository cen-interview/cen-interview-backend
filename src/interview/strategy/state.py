"""Strategy 의 진행 상태.

면접이 진행되는 동안 "지금까지 어떤 주제를, 어떤 난이도로 물었는지"를 유지해
다음 질문이 한 주제에 몰리지 않게 하고 난이도 균형을 맞춘다.
"""
# 최대 꼬리질문 3개

from pydantic import BaseModel, Field

from interview.schemas.question import Difficulty
from interview.schemas.signals import AnswerQuality


class StrategyState(BaseModel):
    """Strategy 의 진행 상태.

    면접이 진행되는 동안 "지금까지 어떤 주제를, 어떤 난이도로 물었는지"를 유지해
    다음 질문이 한 주제에 몰리지 않게 하고 난이도 균형을 맞춘다.

    Attributes:
        asked_topics:
            지금까지 출제된 질문의 주제 목록. 순서대로 쌓인다.

        asked_difficulties:
            지금까지 출제된 질문의 난이도 목록. asked_topics와 인덱스가 대응한다.

        question_count:
            지금까지 출제된 메인 질문 수. 파생 질문(꼬리/압박/확인/함정)은
            포함하지 않는다.
        
        derived_question_count:
            지금까지 출제된 파생 질문(꼬리/압박/확인/함정) 수. hint는
            실제 출제된 질문이 아니므로 포함하지 않는다.

        asked_question_texts:
            지금까지 출제된 질문의 실제 문장 목록. 중복 질문 방지를 위해
            LLM 프롬프트에 "이미 한 질문"으로 전달할 때 사용한다 (4단계).

        topic_last_quality:
            주제별 마지막 답변 평가 결과. next_question() 호출 시에만 갱신된다
            (파생 질문은 답변 평가 신호를 받지 않으므로 갱신하지 않는다).
    """

    asked_topics: list[str] = Field(default_factory=list)
    asked_difficulties: list[Difficulty] = Field(default_factory=list)
    question_count: int = 0
    derived_question_count: int = 0

    asked_question_texts: list[str] = Field(default_factory=list)
    topic_last_quality: dict[str, AnswerQuality] = Field(default_factory=dict)

    def topic_counts(self) -> dict[str, int]:
        """주제별 출제 횟수"""
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
