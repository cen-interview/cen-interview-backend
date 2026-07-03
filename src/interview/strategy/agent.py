"""Strategy Agent.

면접 질문의 방향·순서·난이도를 조정하는 전략 담당.
Interviewer가 전달한 AnswerQualitySignal을 바탕으로 다음 질문 생성을 question_gen에 위임한다.
"""

#from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Question
from interview.schemas.signals import AnswerQualitySignal
from interview.strategy import difficulty, question_gen  # noqa: F401 (TODO 담당 B: question_gen 연결 시 사용)
from interview.strategy.state import StrategyState


class StrategyAgent:
    def __init__(self) -> None:
        self.state = StrategyState()

    def next_question(self, last_signal: AnswerQualitySignal | None) -> Question:
        """
        TODO(담당 B):
          - 아직 적게 다룬 주제 우선 (state.topic_counts)
          - coverage.weak_topics() 는 피하거나 일반 질문으로 대체
          - difficulty.next_difficulty 로 난이도 결정 후 question_gen 호출
          - state 갱신 (asked_topics/difficulties/count)
        """

        """다음 메인 질문 선택."""
        diff = difficulty.next_difficulty(self.state, last_signal)
        topic = self._pick_topic()
        
        # [실전 전환용 주석] question_gen이 완성되면 아래 주석을 풀고 Stub 리턴을 주석 처리하세요.
        # return question_gen.generate_question(topic, diff)
        
        # [현재 Stub 작동]
        text = self._dummy_questions.get(topic, "공통 질문입니다.")
        return Question(
            question_id=f"q_main_{self._topic_idx}",
            text=text,
            topic=topic,
            difficulty=diff,
            kind="main"
        )

    def next_follow_up(self, topic: str, target: str | None = None) -> Question:
        """추가 확인 가능한 요소에 대한 꼬리 질문 생성."""
        return question_gen.generate_follow_up(topic, target)

    def next_challenge(self, topic: str, target: str | None = None) -> Question:
        """오개념이나 논리적 허점을 검증하는 압박 질문 생성."""
        return question_gen.generate_challenge(topic, target)

    def next_confirm_positive(self, topic: str, target: str | None = None) -> Question:
        """답변이 대체로 맞지만 범위나 사실관계를 확인하는 긍정 확인 질문 생성."""
        return question_gen.generate_confirm_positive(topic, target)

    def next_confirm_negative(self, topic: str, target: str | None = None) -> Question:
        """Evidence 또는 이전 답변과 충돌하는 내용을 확인하는 부정 확인 질문 생성."""
        return question_gen.generate_confirm_negative(topic, target)

    def next_trap(self, topic: str, target: str | None = None) -> Question:
        """헷갈리기 쉬운 개념 구분을 확인하는 함정 질문 생성."""
        return question_gen.generate_trap(topic, target)

    def _pick_topic(self) -> str:
        """다음 주제 선택 임시 스텁."""
        return "FastAPI"
