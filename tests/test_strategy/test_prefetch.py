"""next_question()의 다음 질문 프리페치(백그라운드 미리 생성) 동작 테스트.

실제 LLM/그래프를 타면 느리고 네트워크에 의존하므로, StrategyAgent._graph를
FakeGraph로 바꿔치기해 "호출 몇 번 일어났는지"와 "어떤 결과가 쓰였는지"만으로
프리페치 적중/불일치/대기/중복폐기 네 가지 분기를 검증한다.
"""
import time

from interview.schemas.question import Difficulty, Question, QuestionCategory, QuestionKind
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal
from interview.strategy.agent import StrategyAgent

# 백그라운드 프리페치가 끝날 시간을 기다려주는 여유 시간
_SETTLE = 0.05


class FakeGraph:
    """StrategyAgent._graph 대체용 가짜 그래프.

    실제 QuestionGenState 그래프 대신 호출 순서대로 "가짜 질문 N" 텍스트를
    돌려준다. delay를 주면 invoke()가 그 시간만큼 블로킹해 "아직 프리페치가
    끝나지 않은 상황"을 재현할 수 있다.
    """

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list = []
        self.delay = delay

    def invoke(self, state) -> dict:
        if self.delay:
            time.sleep(self.delay)
        index = len(self.calls)
        self.calls.append(state)
        return {
            "result": Question(
                question_id=f"q-{index}",
                text=f"가짜 질문 {index}는 무엇입니까?",
                topic="FastAPI",
                difficulty=state.difficulty,
                kind=QuestionKind.MAIN,
                category=QuestionCategory.TECHNICAL,
            )
        }


def _signal(quality: AnswerQuality) -> AnswerQualitySignal:
    return AnswerQualitySignal(answer_id="a", question_id="q", quality=quality)


def test_matching_guess_serves_prefetched_question_without_extra_call():
    """추정 난이도가 실제와 맞으면 동기 호출 없이 프리페치 결과를 그대로 쓴다."""
    strategy = StrategyAgent()
    fake = FakeGraph()
    strategy._graph = fake

    q1 = strategy.next_question(last_signal=None)
    assert len(fake.calls) == 1  # 첫 질문은 프리페치가 없어 동기 생성

    time.sleep(_SETTLE)  # q1 직후 시작된 "다음 질문" 프리페치가 끝나길 기다림
    assert len(fake.calls) == 2

    q2 = strategy.next_question(last_signal=_signal(AnswerQuality.SUFFICIENT))

    assert len(fake.calls) == 2  # 캐시를 썼으므로 동기 호출이 추가되지 않음
    assert q2.question_id == "q-1"
    assert q1.question_id != q2.question_id


def test_difficulty_guess_mismatch_falls_back_to_sync_generation():
    """추정 난이도가 실제와 어긋나면 프리페치를 버리고 그 자리에서 새로 만든다."""
    strategy = StrategyAgent()
    fake = FakeGraph()
    strategy._graph = fake

    strategy.next_question(last_signal=None)  # EASY
    time.sleep(_SETTLE)
    strategy.next_question(last_signal=_signal(AnswerQuality.SUFFICIENT))  # 여전히 EASY (아직 연속 2회 아님)
    time.sleep(_SETTLE)

    calls_before = len(fake.calls)
    # 직전 두 메인 질문이 모두 EASY라 difficulty.py 규칙 3(연속 2회 EASY -> 강제 상승)이
    # 발동한다. 방금 시작해둔 프리페치는 "직전과 동일(EASY)"로 추정했으므로 실제
    # 난이도(MEDIUM)와 어긋난다.
    q3 = strategy.next_question(last_signal=_signal(AnswerQuality.SUFFICIENT))

    assert q3.difficulty == Difficulty.MEDIUM
    assert len(fake.calls) == calls_before + 1  # 캐시를 버리고 동기 호출 1회 추가


def test_next_question_waits_for_inflight_prefetch_instead_of_duplicating_work():
    """캐시가 아직 진행 중이어도 난이도가 맞으면 새로 만들지 않고 그 결과를 기다린다."""
    strategy = StrategyAgent()
    fake = FakeGraph(delay=0.2)
    strategy._graph = fake

    strategy.next_question(last_signal=None)  # 동기 호출 1회 + 백그라운드 프리페치 시작(0.2초 소요)
    assert len(fake.calls) == 1

    start = time.perf_counter()
    strategy.next_question(last_signal=_signal(AnswerQuality.SUFFICIENT))
    elapsed = time.perf_counter() - start

    assert len(fake.calls) == 2  # 새로 동기 호출을 만들지 않고 기존 프리페치 하나만 사용
    # 새로 0.2초짜리 호출을 또 했다면 0.2초에 근접했을 것 - 먼저 시작해둔 프리페치의
    # 남은 시간만 기다렸는지를 넉넉한 여유를 두고 확인한다.
    assert elapsed < fake.delay * 1.5


def test_stale_prefetch_discarded_when_it_duplicates_a_question_asked_meanwhile():
    """프리페치 시작 이후 같은 문구의 질문이 이미 나왔다면(꼬리질문 등) 캐시를 버린다."""
    strategy = StrategyAgent()
    fake = FakeGraph()
    strategy._graph = fake

    strategy.next_question(last_signal=None)
    time.sleep(_SETTLE)

    # 프리페치가 만들어둔 "가짜 질문 1는 무엇입니까?"와 동일한 문구가 그 사이 다른
    # 경로(꼬리질문 등)로 이미 나왔다고 가정한다.
    strategy.state.asked_question_texts.append("가짜 질문 1는 무엇입니까?")

    calls_before = len(fake.calls)
    strategy.next_question(last_signal=_signal(AnswerQuality.SUFFICIENT))

    assert len(fake.calls) == calls_before + 1  # 캐시를 버리고 새로 동기 생성
