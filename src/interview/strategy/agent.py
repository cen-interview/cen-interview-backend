"""Strategy Agent.

면접 질문의 방향·순서·난이도를 계속 조정하는 전략 담당. LangGraph 노드에서
호출되는 진입점들을 모아둔다. 실제 생성/난이도 로직은 question_gen / difficulty 에.

Interviewer 와의 협업 (설계 문서 시퀀스):
  Interviewer ──"꼬리질문 하나 생성해줘, 부족 키워드는 X"──▶ Strategy.next_follow_up
  Strategy ────"꼬리질문 + 연결 근거"──────────────────────▶ Interviewer
"""

from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Question
from interview.schemas.signals import AnswerQualitySignal
from interview.strategy import difficulty, question_gen
from interview.strategy.state import StrategyState


class StrategyAgent:
    def __init__(self, coverage: CoverageMap) -> None:
        self.state = StrategyState()
        self.coverage = coverage  # 약한 주제 회피/대체에 사용

    def next_question(self, last_signal: AnswerQualitySignal | None) -> Question:
        """다음 일반 질문 선택.

        TODO(담당 B):
          - 아직 적게 다룬 주제 우선 (state.topic_counts)
          - coverage.weak_topics() 는 피하거나 일반 질문으로 대체
          - difficulty.next_difficulty 로 난이도 결정 후 question_gen 호출
          - state 갱신 (asked_topics/difficulties/count)
        """
        diff = difficulty.next_difficulty(self.state, last_signal)
        topic = self._pick_topic()
        question = question_gen.generate_question(topic, diff)
        self._record_question(question)
        return question

    def next_follow_up(self, topic: str, missing_keywords: list[str]) -> Question:
        """Interviewer 요청에 따라 꼬리 질문 생성."""
        question = question_gen.generate_follow_up(topic, missing_keywords)
        self._record_question(question)
        return question

    def next_hint(self, topic: str) -> Question:
        """막힘 상황 힌트성 질문 생성."""
        question = question_gen.generate_hint(topic)
        self._record_question(question)
        return question

    def _pick_topic(self) -> str:
        """다음 주제 선택 (주제 쏠림 방지).

        TODO(담당 B): coverage + 이미 물어본 주제 분포로 다음 주제 결정
        """
        if not self.coverage.topic_confidence:
            return "기술 면접"

        topic_counts = self.state.topic_counts()
        return min(
            self.coverage.topic_confidence,
            key=lambda topic: (
                topic_counts.get(topic, 0),
                -self.coverage.topic_confidence[topic],
            ),
        )

    def _record_question(self, question: Question) -> None:
        self.state.asked_topics.append(question.topic)
        self.state.asked_difficulties.append(question.difficulty)
        self.state.question_count += 1
