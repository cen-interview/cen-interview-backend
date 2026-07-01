from interview.assessment.agent import AssessmentAgent
from interview.assessment.evaluator import judge_answer
from interview.evidence.retrieval import search_evidence
from interview.schemas.question import Difficulty, Question, QuestionKind
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal
from interview.strategy.agent import StrategyAgent


def make_main_question() -> Question:
    return Question(
        question_id="q-1",
        text="FastAPI에서 Depends를 사용하는 이유는?",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=QuestionKind.MAIN,
    )


def make_follow_up_question() -> Question:
    return Question(
        question_id="q-2",
        text="FastAPI 답변에서 핵심 개념 부분을 조금 더 설명해 주세요.",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=QuestionKind.FOLLOW_UP,
        parent_question_id="q-1",
    )


def test_search_evidence_stub_returns_chunk():
    chunks = search_evidence(
        query="FastAPI Depends",
        topic="FastAPI",
    )

    assert len(chunks) >= 1
    assert chunks[0].topic == "FastAPI"


def test_judge_answer_main_returns_bonus_available_stub():
    signal = judge_answer(
        question=make_main_question(),
        answer_text="Depends는 의존성 주입입니다.",
    )

    assert isinstance(signal, AnswerQualitySignal)
    assert signal.quality == AnswerQuality.BONUS_AVAILABLE


def test_judge_answer_follow_up_returns_sufficient_stub():
    signal = judge_answer(
        question=make_follow_up_question(),
        answer_text="DB 세션 같은 공통 의존성을 주입해서 재사용합니다.",
    )

    assert signal.quality == AnswerQuality.SUFFICIENT


def test_strategy_next_question_returns_main_question():
    strategy = StrategyAgent()

    signal = AnswerQualitySignal(
        question_id="q-1",
        quality=AnswerQuality.SUFFICIENT,
    )

    question = strategy.next_question(last_signal=signal)

    assert isinstance(question, Question)
    assert question.kind == QuestionKind.MAIN


def test_strategy_next_follow_up_returns_follow_up_question():
    strategy = StrategyAgent()

    question = strategy.next_follow_up(
        topic="FastAPI",
        target="Depends의 동작 방식",
    )

    assert question.kind == QuestionKind.FOLLOW_UP


def test_strategy_next_challenge_returns_challenge_question():
    strategy = StrategyAgent()

    question = strategy.next_challenge(
        topic="FastAPI",
        target="Depends를 라우터 생성 기능으로 오해한 부분",
    )

    assert question.kind == QuestionKind.CHALLENGE


def test_strategy_next_confirm_positive_returns_confirm_positive_question():
    strategy = StrategyAgent()

    question = strategy.next_confirm_positive(
        topic="FastAPI",
        target="Depends를 실제 프로젝트에서 사용한 범위",
    )

    assert question.kind == QuestionKind.CONFIRM_POSITIVE


def test_strategy_next_confirm_negative_returns_confirm_negative_question():
    strategy = StrategyAgent()

    question = strategy.next_confirm_negative(
        topic="FastAPI",
        target="이전 답변 또는 Evidence와 충돌하는 부분",
    )

    assert question.kind == QuestionKind.CONFIRM_NEGATIVE


def test_strategy_next_trap_returns_trap_question():
    strategy = StrategyAgent()

    question = strategy.next_trap(
        topic="FastAPI",
        target="Depends와 라우터의 역할 차이",
    )

    assert question.kind == QuestionKind.TRAP


def test_assessment_agent_evaluate_returns_signal():
    assessment = AssessmentAgent()

    signal = assessment.evaluate(
        question=make_main_question(),
        answer_text="Depends는 의존성 주입입니다.",
    )

    assert isinstance(signal, AnswerQualitySignal)