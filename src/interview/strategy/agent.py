"""Strategy Agent.

면접 질문의 방향·순서·난이도를 계속 조정하는 전략 담당. LangGraph 노드에서
호출되는 진입점들을 모아둔다. 실제 생성/난이도 로직은 question_gen / difficulty 에.

Interviewer 와의 협업 (설계 문서 시퀀스):
  Interviewer ──"꼬리질문 하나 생성해줘, 부족 키워드는 X"──▶ Strategy.next_follow_up
  Strategy ────"꼬리질문 + 연결 근거"──────────────────────▶ Interviewer
"""

#from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Question
from interview.schemas.signals import AnswerQualitySignal
from interview.strategy import difficulty, question_gen
from interview.strategy.state import StrategyState


class StrategyAgent:
    def __init__(self) -> None:
        self.state = StrategyState()

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
        return question_gen.generate_question(topic, diff)

    def next_follow_up(self, topic: str, missing_keywords: list[str]) -> Question:
        """Interviewer 요청에 따라 꼬리 질문 생성. (누락 요소) """
        return question_gen.generate_follow_up(topic, missing_keywords)


    def next_confirm(
        self,
        topic: str,
        misconception_note: str | None = None,
        ) -> Question:
            """오개념이 의심될 때 확인 질문 생성."""
            return question_gen.generate_confirm(topic, misconception_note)

    def _pick_topic(self) -> str:
        """다음 주제 선택 (주제 쏠림 방지).

        TODO(담당 B): coverage + 이미 물어본 주제 분포로 다음 주제 결정
        """
        return "FastAPI"
