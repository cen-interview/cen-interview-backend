"""Interviewer 라우팅 테스트."""

from interview.interviewer.agent import InterviewerAgent
from interview.interviewer.session import SessionState
from interview.schemas.events import AnswerSubmitted
from interview.schemas.question import Question


class FakeAssessment:
    def __init__(self, signal):
        self._signal = signal

    def evaluate(self, **kwargs):
        return self._signal


class FakeStrategy:
    def next_follow_up(self, topic, missing_keywords):
        return Question(
            question_id="follow-1",
            text=f"{topic}에서 {', '.join(missing_keywords)}를 더 설명해주세요.",
            topic=topic,
            difficulty="medium",
            kind="follow_up",
        )


def test_shallow_answer_routes_to_follow_up(sample_question, shallow_signal):
    session = SessionState(
        session_id="s1",
        mode="chat",
        current_question=sample_question,
    )
    interviewer = InterviewerAgent(session, FakeStrategy(), FakeAssessment(shallow_signal))

    question = interviewer.handle(
        AnswerSubmitted(
            session_id="s1",
            question_id=sample_question.question_id,
            text="짧은 답변",
        )
    )

    assert question.kind == "follow_up"
    assert session.current_question == question
