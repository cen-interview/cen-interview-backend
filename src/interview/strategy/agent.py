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
        self._topic_idx = 0
        self._dummy_questions: dict[str, str] = {
            "FastAPI": "FastAPI에서 Depends를 사용하는 이유는 무엇인가요?",
        }

    def next_question(self, last_signal: AnswerQualitySignal | None) -> Question:
        """
        TODO(담당 B):
          - 아직 적게 다룬 주제 우선 (state.topic_counts)
          - coverage.weak_topics() 는 피하거나 일반 질문으로 대체
          - difficulty.next_difficulty 로 난이도 결정 후 question_gen 호출
          - state 갱신 (asked_topics/difficulties/count)
        """

        """다음 메인 질문을 생성한다.

        Interviewer가 새로운 주제로 넘어갈 때 호출한다. 직전 답변 평가(last_signal)를
        참고해 난이도를 조정하고, 아직 다루지 않은 주제를 우선 선택한다.

        Args:
        last_signal: 직전 질문에 대한 답변 평가 결과.
            첫 질문이라 이전 답변이 없으면 None을 전달한다.

        Returns:
            kind=MAIN인 Question.

        Side effect:
            호출 시마다 self.state(asked_topics/asked_difficulties/question_count)가
            갱신된다.
        """
        diff = difficulty.next_difficulty(self.state, last_signal)
        topic = self._pick_topic()
        
        # [실전 전환용 주석] question_gen이 완성되면 아래 주석을 풀고 Stub 리턴을 주석 처리하세요.
        # return question_gen.generate_question(topic, diff)
        
        # [현재 Stub 작동]
        text = self._dummy_questions.get(topic, "공통 질문입니다.")
        question = Question(
            question_id=f"q_main_{self._topic_idx}",
            text=text,
            topic=topic,
            difficulty=diff,
            kind="main"
        )

        self.state.asked_topics.append(topic)
        self.state.asked_difficulties.append(diff)
        self.state.question_count += 1

        return question

    def next_follow_up(
        self, 
        topic: str, 
        parent_question_id: str, 
        target: str | None = None,
        answer_excerpt: str | None = None,
        ) -> Question:
        """추가 확인 가능한 요소에 대한 꼬리 질문 생성.

        Args:
        topic: 질문 주제.
        parent_question_id: 이 꼬리 질문이 파생된 원래 메인 질문의 ID.
        target: 무엇을 더 캐물을지 (예: "Depends의 동작 방식").
        answer_excerpt: 사용자의 직전 답변 중 인용할 부분 (선택).
            Interviewer가 transcript에서 짧게(핵심 문장 1~2개) 잘라 전달한다.
            제공되면 "방금 ~라고 하셨는데" 형태로 답변을 직접 인용하는 질문을 만든다.
        """
        return question_gen.generate_follow_up(topic, parent_question_id, target, answer_excerpt)

    def next_challenge(
        self, 
        topic: str,
        parent_question_id: str, 
        target: str | None = None,
        answer_excerpt: str | None = None,
        ) -> Question:
        """오개념이나 논리적 허점을 검증하는 압박 질문 생성.
        
        Args:
        topic: 질문 주제.
        parent_question_id: 이 꼬리 질문이 파생된 원래 메인 질문의 ID.
        target: 무엇을 더 캐물을지 (예: "Depends의 동작 방식").
        answer_excerpt: 사용자의 직전 답변 중 인용할 부분 (선택).
            Interviewer가 transcript에서 짧게(핵심 문장 1~2개) 잘라 전달한다.
            제공되면 "방금 ~라고 하셨는데" 형태로 답변을 직접 인용하는 질문을 만든다.
        """
        return question_gen.generate_challenge(topic, parent_question_id, target, answer_excerpt)

    def next_confirm_positive(
            self, 
            topic: str, 
            parent_question_id: str, 
            target: str | None = None,
            answer_excerpt: str | None = None
            ) -> Question:
        """답변이 대체로 맞지만 범위나 사실관계를 확인하는 긍정 확인 질문 생성."""
        return question_gen.generate_confirm_positive(topic, parent_question_id, target, answer_excerpt)

    def next_confirm_negative(
        self, 
        topic: str, 
        parent_question_id: str, 
        target: str | None = None,
        answer_excerpt: str | None = None
        ) -> Question:
        """Evidence 또는 이전 답변과 충돌하는 내용을 확인하는 부정 확인 질문 생성."""
        return question_gen.generate_confirm_negative(topic, parent_question_id, target, answer_excerpt)

    def next_trap(
        self, 
        topic: str, 
        parent_question_id: str, 
        target: str | None = None,
        answer_excerpt: str | None = None
        ) -> Question:
        """헷갈리기 쉬운 개념 구분을 확인하는 함정 질문 생성."""
        return question_gen.generate_trap(topic, parent_question_id, target, answer_excerpt)

    def next_hint(
        self, 
        question: Question, 
        target: str | None = None,
        answer_excerpt: str | None = None
        ) -> Question:
        """침묵 등으로 사용자가 답변을 못 할 때 호출하는 힌트 생성.

        정답을 알려주지 않고 접근 방향만 제시한다 (6단계에서 구현 예정).

        Args:
            question: 힌트를 줄 대상이 되는 원래 질문.
            target: 힌트를 어느 부분에 집중할지 (선택).
            answer_excerpt: 사용자의 직전 답변 중 인용할 부분 (선택).
                완전 침묵이면 None. 답변은 했지만 방향이 틀린 경우 참고용으로 전달.

        Returns:
            kind=HINT인 Question. parent_question_id는 원래 question의 ID.
        """
        return question_gen.generate_hint(question, target, answer_excerpt)

    def _pick_topic(self) -> str:
        """다음 주제 선택 임시 스텁."""
        return "FastAPI"
