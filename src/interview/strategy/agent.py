"""Strategy Agent.

면접 질문의 방향·순서·난이도를 조정하는 전략 담당.
Interviewer가 전달한 AnswerQualitySignal을 바탕으로 다음 질문 생성을 question_gen에 위임한다.
"""

from concurrent.futures import Future, ThreadPoolExecutor

from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Difficulty, Question, QuestionKind
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal
from interview.strategy import difficulty, question_gen  # noqa: F401 (TODO 담당 B: question_gen 연결 시 사용)
from interview.strategy.graph import QuestionGenState, _too_similar, get_compiled_graph
from interview.strategy.state import StrategyState

# 프리페치 난이도 추정에 쓰는 "중립" quality. difficulty.next_difficulty()의 규칙 중
# 이 값으로는 걸리지 않는 것(MISCONCEPTION/CONFIRM_NEGATIVE/UNKNOWN 하강, 연속
# SUFFICIENT 상승)만 실제 last_signal이 나와야 아는 것이고, 나머지 규칙(연속 EASY
# 강제 상승, N문항 이상 HARD 강제)은 last_signal 없이 state만으로도 이미 확정되므로
# 이 값으로 next_difficulty()를 그대로 호출하면 정확히 맞힌다.
# UNKNOWN은 더 이상 중립이 아니다(하강 트리거로 승격됨) - BONUS_AVAILABLE을 쓴다.
_NEUTRAL_GUESS_QUALITY = AnswerQuality.BONUS_AVAILABLE


class StrategyAgent:
    """면접 질문의 방향·순서·난이도를 결정하는 전략 담당 에이전트.

    Interviewer로부터 답변 평가 신호(AnswerQualitySignal)를 받아 다음 질문을
    결정하고, 실제 질문 문장 생성은 question_gen 모듈에 위임한다. 세션 동안의
    출제 이력은 self.state(StrategyState)에 누적된다.

    Attributes:
        state:
            세션 동안 누적되는 출제 이력 (주제, 난이도, 질문 수 등).

        weak_history_topics:
            이전 면접 이력에서 약점으로 평가된 주제 목록. 세션 시작 시
            호출자(facade.create_session)가 한 번 조회해 생성자로 넘겨주는
            고정값이라 self에 저장해두고 매 next_question() 호출마다
            QuestionGenState에 그대로 전달한다.

        _topic_idx:
            [임시] stub 질문 ID 생성에 사용하는 인덱스. question_gen.generate_question 연결 시 제거될 예정.

        _dummy_questions:
            [임시] stub 질문 텍스트 매핑. 제거 예정.

        _executor:
            메인 질문 프리페치(다음 질문 미리 생성)를 실행하는 단일 워커
            스레드 풀. 세션 하나당 프리페치는 한 번에 하나만 진행하면 되므로
            worker 1개로 충분하다.

        _prefetch_future:
            진행 중이거나 완료된 프리페치 결과. next_question()이 매번 자신의
            응답을 반환하기 직전에 다음 질문 프리페치를 새로 시작하며 이
            값을 덮어쓴다. 없으면 None.

        _prefetch_difficulty:
            프리페치를 시작할 때 사용한 "추정 난이도". 직전 질문과 같은
            난이도를 그대로 재사용한다 (difficulty.next_difficulty의 기본
            분기가 "직전 난이도 유지"이므로 대부분 들어맞는다). 실제
            next_question() 호출에서 계산된 진짜 난이도와 비교해 일치할
            때만 프리페치 결과를 사용한다.
    """

    def __init__(
        self,
        coverage: CoverageMap | None=None,
        user_id: str | None=None,
        weak_history_topics: list[str] | None=None,
    ) -> None:
        self.state = StrategyState()
        self.coverage = coverage or CoverageMap()
        self.user_id = user_id
        self.weak_history_topics = weak_history_topics or []
        self._graph = get_compiled_graph()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_future: Future | None = None
        self._prefetch_difficulty: Difficulty | None = None

    def next_question(self, last_signal: AnswerQualitySignal | None) -> Question:
        """다음 메인 질문을 생성한다.

        Interviewer가 새로운 주제로 넘어갈 때 호출한다. 직전 답변 평가(last_signal)를
        참고해 난이도를 조정하고, 내부적으로 질문 생성 그래프(graph.py)를 통해
        주제 선택부터 검증까지 처리한다.

        직전 호출에서 미리 생성해둔(prefetch) 질문이 있고 그때 추정한 난이도가
        지금 계산된 실제 난이도와 일치하면 그 결과를 그대로 쓴다 (동기 LLM 호출
        생략). 추정이 빗나갔거나 아직 준비되지 않았으면 기존처럼 그 자리에서
        동기 생성한다 - 이 경우도 기존 대비 느려지지 않는다.

        Args:
        last_signal: 직전 질문에 대한 답변 평가 결과.
            첫 질문이라 이전 답변이 없으면 None을 전달한다.

        Returns:
            kind=MAIN인 Question.

        Side effect:
        호출 시마다 self.state가 갱신된다. last_signal이 있으면
        topic_last_quality, recent_qualities도 함께 갱신된다. 반환 직전에
        다음 질문에 대한 프리페치를 새로 시작한다.
        """
        diff = difficulty.next_difficulty(self.state, last_signal)

        question = self._consume_prefetch(diff)
        if question is None:
            initial_state = QuestionGenState(
                coverage=self.coverage,
                strategy_state=self.state,
                difficulty=diff,
                user_id=self.user_id,
                weak_history_topics=self.weak_history_topics,
            )
            result_state = self._graph.invoke(initial_state)
            question = result_state["result"]

        if last_signal is not None:
            self.state.topic_last_quality[question.topic] = last_signal.quality
            self.state.recent_qualities.append(last_signal.quality)

        self._record(question)
        self._start_prefetch(guessed_difficulty=self._guess_next_difficulty())

        return question

    def _guess_next_difficulty(self) -> Difficulty:
        """다음 메인 질문의 난이도를 프리페치 시점에 최대한 정확히 추정한다.

        difficulty.next_difficulty()의 규칙 중 "연속 2회 EASY 강제 상승"과
        "N문항 이상인데 HARD 없음 강제 상승"은 last_signal 없이 self.state만
        보고도 이미 확정된다 - 이 두 규칙이 실전에서 프리페치 불일치의 주된
        원인이었다. 아직 알 수 없는 건 "오개념/부정확인/모름 하강"과 "연속 2회
        SUFFICIENT 상승" 뿐이므로, 이들에는 걸리지 않는 중립 quality
        (_NEUTRAL_GUESS_QUALITY)로 실제 next_difficulty()를 그대로 호출해 앞의
        두 규칙은 정확히 맞히고 나머지는 기존처럼 "직전 난이도 유지"로 추정한다.
        """
        placeholder_signal = AnswerQualitySignal(
            answer_id="_prefetch_guess",
            question_id="_prefetch_guess",
            quality=_NEUTRAL_GUESS_QUALITY,
        )
        return difficulty.next_difficulty(self.state, placeholder_signal)

    def _start_prefetch(self, guessed_difficulty: Difficulty) -> None:
        """다음 메인 질문 생성을 백그라운드 스레드에서 미리 시작한다.

        지원자가 현재 질문에 답하는 동안(다음 next_question() 호출까지의
        유휴 시간) 미리 LLM 생성을 끝내두기 위함이다. guessed_difficulty는
        _guess_next_difficulty()가 계산한 추정값이다.

        self.state를 백그라운드 스레드가 그대로 참조하면 그 사이 파생 질문
        기록(_record) 등으로 메인 스레드가 동시에 수정할 수 있어 deep copy로
        스냅샷을 떠서 넘긴다.
        """
        state_snapshot = self.state.model_copy(deep=True)
        initial_state = QuestionGenState(
            coverage=self.coverage,
            strategy_state=state_snapshot,
            difficulty=guessed_difficulty,
            user_id=self.user_id,
            weak_history_topics=self.weak_history_topics,
        )
        self._prefetch_difficulty = guessed_difficulty
        self._prefetch_future = self._executor.submit(self._graph.invoke, initial_state)

    def _consume_prefetch(self, diff: Difficulty) -> Question | None:
        """준비된 프리페치 결과가 지금 필요한 난이도와 맞으면 꺼내 쓴다.

        난이도가 다르면(추정이 빗나간 경우) 그냥 버리고 None을 반환해
        next_question()이 평소처럼 동기 생성하게 한다 - 아직 실행 중인
        스레드가 있어도 취소하지 않고 결과만 버린다(완료되면 조용히 소멸).

        난이도가 맞으면 future가 끝나 있지 않아도 기다린다 - 이미 먼저
        시작된 작업이므로 지금 새로 시작하는 것보다 항상 빠르거나 같다.

        꺼내 쓰기 직전에 이미 나온 질문과 중복인지 한 번 더(LLM 호출 없이)
        검사한다 - 프리페치 시작 이후 꼬리 질문 등으로 asked_question_texts가
        늘어났을 수 있어 프리페치 시점의 validate() 결과만으로는 부족하다.
        """
        future = self._prefetch_future
        guessed_difficulty = self._prefetch_difficulty
        self._prefetch_future = None
        self._prefetch_difficulty = None

        if future is None or guessed_difficulty != diff:
            return None

        try:
            result_state = future.result()
        except Exception:
            return None

        question = result_state.get("result")
        if question is None:
            return None
        if any(
            _too_similar(question.text, asked)
            for asked in self.state.asked_question_texts
        ):
            return None

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

    def close(self) -> None:
        """프리페치 executor와 대기 중인 프리페치 작업을 정리한다.

        세션 종료 시 facade의 세션 정리 경로에서 호출한다. 이미 실행 중인
        프리페치 LLM 호출은 중단할 수 없지만, 큐에서 대기 중인 작업은
        취소하고 worker 스레드가 유휴 상태로 남지 않게 executor를 닫는다.
        """
        self._prefetch_future = None
        self._prefetch_difficulty = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _record(self, question: Question) -> None:
        """질문 생성 후 state를 한 곳에서 갱신한다 (hint 제외 모든 next_*가 호출)."""
        self.state.asked_topics.append(question.topic)
        self.state.asked_difficulties.append(question.difficulty)
        self.state.asked_question_texts.append(question.text)
        if question.kind == QuestionKind.MAIN:
            self.state.question_count += 1
        else:
            self.state.derived_question_count += 1
