"""InterviewerAgent 라우팅 단위 테스트.

Interviewer는 답변을 직접 평가하거나 질문 문장을 직접 만들지 않는다.
역할은 "이벤트를 받고, Assessment가 준 AnswerQualitySignal을 보고,
Strategy의 어떤 메서드를 호출할지 결정한 뒤 SessionState를 갱신하는 것"이다.

그래서 이 테스트는 실제 AssessmentAgent / StrategyAgent를 쓰지 않고
FakeAssessment / FakeStrategy를 주입한다. 이렇게 해야 evaluator 랜덤값,
evidence 검색, LLM 질문 생성, scoring/report 구현 변화와 무관하게
Interviewer의 책임만 독립적으로 확인할 수 있다.
"""

import pytest

from interview.interviewer import InterviewerAgent, SessionState
from interview.schemas.events import (
    AnswerSubmitted,
    EndRequested,
    NoResponseTimeout,
    ReplayRequested,
    SilenceDetected,
)
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


def make_question(
    question_id: str = "q-main",
    kind: QuestionKind = QuestionKind.MAIN,
    topic: str = "FastAPI",
    parent_question_id: str | None = None,
) -> Question:
    """Interviewer 테스트에서 사용할 최소 Question fixture를 만든다."""
    return Question(
        question_id=question_id,
        text=f"{topic} 테스트 질문",
        topic=topic,
        difficulty=Difficulty.EASY,
        kind=kind,
        category=QuestionCategory.TECHNICAL,
        parent_question_id=parent_question_id,
    )


class FakeAssessment:
    """Interviewer가 Assessment와 맺는 계약만 흉내 내는 fake.

    evaluate()는 테스트에서 지정한 quality를 가진 AnswerQualitySignal을
    고정으로 반환한다. 실제 채점, evidence 검색, report 생성은 검증 대상이
    아니므로 의도적으로 구현하지 않는다.
    """

    def __init__(self, quality: AnswerQuality) -> None:
        self.quality = quality
        self.evaluate_calls = []
        self.completed_main_question_ids = []

    def evaluate(
        self,
        question: Question,
        answer_text: str,
        delivery_metrics: dict | None = None,
    ) -> AnswerQualitySignal:
        self.evaluate_calls.append(
            {
                "question": question,
                "answer_text": answer_text,
                "delivery_metrics": delivery_metrics,
            }
        )
        return AnswerQualitySignal(
            answer_id="answer-1",
            question_id=question.question_id,
            quality=self.quality,
            next_probe_target="probe",
            rationale=["fake assessment"],
        )

    def complete_question_set(self, main_question_id: str) -> None:
        """SUFFICIENT 흐름에서 질문 세트 완료 요청이 왔는지만 기록한다."""
        self.completed_main_question_ids.append(main_question_id)


class FakeStrategy:
    """Interviewer가 Strategy와 맺는 라우팅 계약만 기록하는 fake.

    각 next_* 메서드는 어떤 메서드가 어떤 인자로 호출됐는지 calls에 남기고,
    해당 kind의 고정 Question을 반환한다. 실제 질문 생성 품질은 여기서
    확인하지 않는다.
    """

    def __init__(self) -> None:
        self.calls = []

    def next_question(self, last_signal: AnswerQualitySignal | None) -> Question:
        self.calls.append(("next_question", last_signal))
        return make_question(question_id="q-next-main", kind=QuestionKind.MAIN, topic="Docker")

    def next_follow_up(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        self.calls.append(("next_follow_up", topic, parent_question_id, target, answer_excerpt))
        return make_question("q-follow-up", QuestionKind.FOLLOW_UP, topic, parent_question_id)

    def next_challenge(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        self.calls.append(("next_challenge", topic, parent_question_id, target, answer_excerpt))
        return make_question("q-challenge", QuestionKind.CHALLENGE, topic, parent_question_id)

    def next_confirm_positive(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        self.calls.append(("next_confirm_positive", topic, parent_question_id, target, answer_excerpt))
        return make_question("q-confirm-positive", QuestionKind.CONFIRM_POSITIVE, topic, parent_question_id)

    def next_confirm_negative(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        self.calls.append(("next_confirm_negative", topic, parent_question_id, target, answer_excerpt))
        return make_question("q-confirm-negative", QuestionKind.CONFIRM_NEGATIVE, topic, parent_question_id)

    def next_trap(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        self.calls.append(("next_trap", topic, parent_question_id, target, answer_excerpt))
        return make_question("q-trap", QuestionKind.TRAP, topic, parent_question_id)


def make_session(
    current_question: Question | None = None,
    asked_count: int = 1,
    max_questions: int = 10,
) -> SessionState:
    """현재 질문이 하나 제시된 상태의 세션을 만든다."""
    question = current_question or make_question()
    return SessionState(
        session_id="session-1",
        current_question=question,
        asked_count=asked_count,
        max_questions=max_questions,
        main_question_id=question.question_id,
        main_topic=question.topic,
    )


@pytest.mark.parametrize(
    ("quality", "expected_call", "expected_kind"),
    [
        (AnswerQuality.BONUS_AVAILABLE, "next_follow_up", QuestionKind.FOLLOW_UP),
        (AnswerQuality.MISCONCEPTION, "next_challenge", QuestionKind.CHALLENGE),
        (AnswerQuality.CONFIRM_POSITIVE, "next_confirm_positive", QuestionKind.CONFIRM_POSITIVE),
        (AnswerQuality.CONFIRM_NEGATIVE, "next_confirm_negative", QuestionKind.CONFIRM_NEGATIVE),
        (AnswerQuality.TRAP_AVAILABLE, "next_trap", QuestionKind.TRAP),
    ],
)
def test_answer_quality_routes_to_derived_question(
    quality: AnswerQuality,
    expected_call: str,
    expected_kind: QuestionKind,
):
    """추가 검증이 필요한 quality는 알맞은 파생 질문 생성 메서드로 라우팅된다."""
    session = make_session()
    assessment = FakeAssessment(quality)
    strategy = FakeStrategy()
    interviewer = InterviewerAgent(session, strategy, assessment)

    question = interviewer.handle(
        AnswerSubmitted(
            session_id=session.session_id,
            question_id=session.current_question.question_id,
            text="테스트 답변입니다.",
        )
    )

    assert question is not None
    assert question.kind == expected_kind
    assert question.parent_question_id == "q-main"
    assert session.current_question == question
    assert session.asked_count == 1
    assert strategy.calls[0][0] == expected_call
    assert strategy.calls[0][3] == "probe"
    assert assessment.completed_main_question_ids == []


def test_sufficient_answer_completes_set_and_routes_to_next_main_question():
    """충분한 답변이면 현재 질문 세트를 완료하고 다음 메인 질문으로 넘어간다."""
    session = make_session()
    assessment = FakeAssessment(AnswerQuality.SUFFICIENT)
    strategy = FakeStrategy()
    interviewer = InterviewerAgent(session, strategy, assessment)

    question = interviewer.handle(
        AnswerSubmitted(
            session_id=session.session_id,
            question_id=session.current_question.question_id,
            text="충분한 답변입니다.",
        )
    )

    assert question is not None
    assert question.kind == QuestionKind.MAIN
    assert session.current_question == question
    assert session.asked_count == 2
    assert session.main_question_id == "q-next-main"
    assert session.main_topic == "Docker"
    assert assessment.completed_main_question_ids == ["q-main"]
    assert strategy.calls[0][0] == "next_question"


def test_sufficient_answer_finishes_when_main_question_limit_is_reached():
    """충분한 답변 후 목표 메인 질문 수에 도달했다면 새 질문 없이 세션을 종료한다."""
    session = make_session(asked_count=1, max_questions=1)
    assessment = FakeAssessment(AnswerQuality.SUFFICIENT)
    strategy = FakeStrategy()
    interviewer = InterviewerAgent(session, strategy, assessment)

    question = interviewer.handle(
        AnswerSubmitted(
            session_id=session.session_id,
            question_id=session.current_question.question_id,
            text="마지막 답변입니다.",
        )
    )

    assert question is None
    assert session.finished is True
    assert assessment.completed_main_question_ids == ["q-main"]
    assert strategy.calls == []


def test_replay_and_silence_keep_current_question():
    """다시 듣기와 침묵 이벤트는 평가 없이 현재 질문을 그대로 다시 제시한다."""
    session = make_session()
    interviewer = InterviewerAgent(
        session,
        FakeStrategy(),
        FakeAssessment(AnswerQuality.SUFFICIENT),
    )

    replayed = interviewer.handle(ReplayRequested(session_id=session.session_id))
    represented = interviewer.handle(
        SilenceDetected(
            session_id=session.session_id,
            silence_duration_seconds=8.4,
        )
    )

    assert replayed == session.current_question
    assert represented == session.current_question
    assert session.finished is False


@pytest.mark.parametrize(
    "event",
    [
        EndRequested(session_id="session-1"),
        NoResponseTimeout(session_id="session-1", elapsed_seconds=30.0),
    ],
)
def test_end_events_finish_session(event):
    """명시적 종료와 무응답 타임아웃 이벤트는 세션을 종료한다."""
    session = make_session()
    interviewer = InterviewerAgent(
        session,
        FakeStrategy(),
        FakeAssessment(AnswerQuality.SUFFICIENT),
    )

    question = interviewer.handle(event)

    assert question is None
    assert session.finished is True
