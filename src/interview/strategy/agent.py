"""Strategy Agent.

면접 질문의 방향·순서·난이도를 조정하는 전략 담당.
Interviewer가 전달한 AnswerQualitySignal을 바탕으로 다음 질문 생성을 question_gen에 위임한다.
"""

from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Question, QuestionKind
from interview.schemas.signals import AnswerQualitySignal
from interview.strategy import difficulty, question_gen  # noqa: F401 (TODO 담당 B: question_gen 연결 시 사용)
from interview.strategy.graph import QuestionGenState, get_compiled_graph
from interview.strategy.state import StrategyState


class StrategyAgent:
    """면접 질문의 방향·순서·난이도를 결정하는 전략 담당 에이전트.

    Interviewer로부터 답변 평가 신호(AnswerQualitySignal)를 받아 다음 질문을
    결정하고, 실제 질문 문장 생성은 question_gen 모듈에 위임한다. 세션 동안의
    출제 이력은 self.state(StrategyState)에 누적된다.

    Attributes:
        state:
            세션 동안 누적되는 출제 이력 (주제, 난이도, 질문 수 등).

        _topic_idx:
            [임시] stub 질문 ID 생성에 사용하는 인덱스. question_gen.generate_question 연결 시 제거될 예정.

        _dummy_questions:
            [임시] stub 질문 텍스트 매핑. 제거 예정.
    """

    def __init__(
        self,
        coverage: CoverageMap | None=None,
        user_id: str | None=None,
    ) -> None:
        self.state = StrategyState()
        self.coverage = coverage or CoverageMap()
        self.user_id = user_id
        self._graph = get_compiled_graph()

    def next_question(self, last_signal: AnswerQualitySignal | None) -> Question:
        """다음 메인 질문을 생성한다.

        Interviewer가 새로운 주제로 넘어갈 때 호출한다. 직전 답변 평가(last_signal)를
        참고해 난이도를 조정하고, 내부적으로 질문 생성 그래프(graph.py)를 통해
        주제 선택부터 검증까지 처리한다

        Args:
        last_signal: 직전 질문에 대한 답변 평가 결과.
            첫 질문이라 이전 답변이 없으면 None을 전달한다.

        Returns:
            kind=MAIN인 Question.

        Side effect:
        호출 시마다 self.state가 갱신된다. last_signal이 있으면
        topic_last_quality, recent_qualities도 함께 갱신된다.
        """
        diff = difficulty.next_difficulty(self.state, last_signal)

        initial_state = QuestionGenState(
            coverage=self.coverage,
            strategy_state=self.state,
            difficulty=diff,
            user_id=self.user_id,
        )
        result_state = self._graph.invoke(initial_state)
        question = result_state["result"]
        
        if last_signal is not None:
            self.state.topic_last_quality[question.topic] = last_signal.quality
            self.state.recent_qualities.append(last_signal.quality)

        self._record(question)

        return question

    def next_follow_up(
        self, 
        topic: str, 
        parent_question_id: str, 
        target: str | None = None,
        answer_excerpt: str | None = None,
        rationale: list[str] | None = None,
        ) -> Question:
        """추가 확인 가능한 요소에 대한 꼬리 질문 생성.

        Args:
        topic: 질문 주제.
        parent_question_id: 이 꼬리 질문이 파생된 원래 메인 질문의 ID.
        target: 무엇을 더 캐물을지 (예: "Depends의 동작 방식").
        answer_excerpt: 사용자의 직전 답변 중 인용할 부분 (선택).
            Interviewer가 transcript에서 짧게(핵심 문장 1~2개) 잘라 전달한다.
            제공되면 "방금 ~라고 하셨는데" 형태로 답변을 직접 인용하는 질문을 만든다.
        rationale: Assessment가 이 파생 질문이 필요하다고 판단한 이유
            (AnswerQualitySignal.rationale). 제공되면 프롬프트에 반영해
            더 정확히 문제 지점을 겨냥한 질문을 만든다.
        """

        question = question_gen.generate_follow_up(topic, parent_question_id, target, answer_excerpt, rationale, self.user_id)
        self._record(question)
        return question
    
    def next_challenge(
        self, 
        topic: str,
        parent_question_id: str, 
        target: str | None = None,
        answer_excerpt: str | None = None,
        rationale: list[str] | None = None,
        ) -> Question:
        """오개념이나 논리적 허점을 검증하는 압박 질문 생성.
        
        Args:
        topic: 질문 주제.
        parent_question_id: 이 꼬리 질문이 파생된 원래 메인 질문의 ID.
        target: 무엇을 더 캐물을지 (예: "Depends의 동작 방식").
        answer_excerpt: 사용자의 직전 답변 중 인용할 부분 (선택).
            Interviewer가 transcript에서 짧게(핵심 문장 1~2개) 잘라 전달한다.
            제공되면 "방금 ~라고 하셨는데" 형태로 답변을 직접 인용하는 질문을 만든다.
        rationale: Assessment가 이 파생 질문이 필요하다고 판단한 이유
            (AnswerQualitySignal.rationale). 제공되면 프롬프트에 반영해
            더 정확히 문제 지점을 겨냥한 질문을 만든다.
        """
        question = question_gen.generate_challenge(topic, parent_question_id, target, answer_excerpt, rationale, self.user_id)
        self._record(question)
        return question

    def next_confirm_positive(
            self, 
            topic: str, 
            parent_question_id: str, 
            target: str | None = None,
            answer_excerpt: str | None = None,
            rationale: list[str] | None = None,
            ) -> Question:
        """답변이 대체로 맞지만 범위나 사실관계를 확인하는 긍정 확인 질문 생성."""
        question = question_gen.generate_confirm_positive(topic, parent_question_id, target, answer_excerpt, rationale, self.user_id)
        self._record(question)
        return question

    def next_confirm_negative(
        self, 
        topic: str, 
        parent_question_id: str, 
        target: str | None = None,
        answer_excerpt: str | None = None,
        rationale: list[str] | None = None,
        ) -> Question:
        """Evidence 또는 이전 답변과 충돌하는 내용을 확인하는 부정 확인 질문 생성."""
        question = question_gen.generate_confirm_negative(topic, parent_question_id, target, answer_excerpt, rationale, self.user_id)
        self._record(question)
        return question
    
    def next_trap(
        self, 
        topic: str, 
        parent_question_id: str, 
        target: str | None = None,
        answer_excerpt: str | None = None,
        rationale: list[str] | None = None,
        ) -> Question:
        """헷갈리기 쉬운 개념 구분을 확인하는 함정 질문 생성."""
        question = question_gen.generate_trap(topic, parent_question_id, target, answer_excerpt, rationale, self.user_id)
        self._record(question)
        return question

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
        return question_gen.generate_hint(question, target, answer_excerpt, self.user_id)

    def _record(self, question: Question) -> None:
        """질문 생성 후 state를 한 곳에서 갱신한다 (hint 제외 모든 next_*가 호출)."""
        self.state.asked_topics.append(question.topic)
        self.state.asked_difficulties.append(question.difficulty)
        self.state.asked_question_texts.append(question.text)
        if question.kind == QuestionKind.MAIN:
            self.state.question_count += 1
        else:
            self.state.derived_question_count += 1
