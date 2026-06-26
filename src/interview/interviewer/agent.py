"""Interviewer Agent.

공통 이벤트 + Assessment 평가 신호 + 세션 상태를 보고 "지금 면접을 어떻게
진행할지" 결정한다. 답변 품질을 직접 평가하지 않는다 (그건 Assessment).

흐름 (설계 7-V / 7-C 공통):
  answer_submitted → Assessment 평가 → 신호의 quality 로 라우팅
    SUFFICIENT → 다음 질문   (Strategy.next_question)
    SHALLOW    → 꼬리 질문   (Strategy.next_follow_up)
    STUCK      → 힌트 질문   (Strategy.next_hint)
    CONFLICT   → 확인 질문
  silence_detected     → 막힘으로 보고 힌트 질문
  end_requested        → 종료
  no_response_timeout  → 우아하게 일시정지/종료 (잠들기 대비)
"""

from interview.assessment import AssessmentAgent
from interview.schemas.events import (
    AnswerSubmitted,
    EndRequested,
    InterviewEvent,
    NoResponseTimeout,
    ReplayRequested,
    SilenceDetected,
)
from interview.schemas.question import Question
from interview.schemas.signals import QualityLevel
from interview.interviewer.session import SessionState
from interview.strategy import StrategyAgent


class InterviewerAgent:
    def __init__(
        self,
        session: SessionState,
        strategy: StrategyAgent,
        assessment: AssessmentAgent,
    ) -> None:
        self.session = session
        self.strategy = strategy
        self.assessment = assessment

    def handle(self, event: InterviewEvent) -> Question | None:
        """이벤트 1건을 처리하고 사용자에게 제시할 다음 질문을 반환한다.
        종료면 None 을 반환하고 session.finished 를 세운다.

        TODO(담당 C): 아래 분기를 채운다. 핵심은 '한 벌의 흐름 로직'을 유지하는 것.
        """
        if isinstance(event, AnswerSubmitted):
            return self._on_answer(event)
        if isinstance(event, SilenceDetected):
            return self._on_stuck()
        if isinstance(event, ReplayRequested):
            return self.session.current_question  # 같은 질문 재제시(TTS 재생)
        if isinstance(event, EndRequested):
            self.session.finished = True
            return None
        if isinstance(event, NoResponseTimeout):
            self.session.finished = True  # 잠들기 대비: 우아하게 종료
            return None
        raise ValueError(f"unhandled event: {event}")

    def _on_answer(self, event: AnswerSubmitted) -> Question | None:
        """답변 처리: Assessment 에 평가 위임 후 신호로 라우팅."""
        signal = self.assessment.evaluate(
            question=self.session.current_question,
            answer_text=event.answer_text,
            delivery_metrics=event.delivery_metrics,
        )
        if self.session.is_done():
            self.session.finished = True
            return None

        topic = self.session.current_question.topic
        if signal.quality == QualityLevel.SHALLOW:
            q = self.strategy.next_follow_up(topic, signal.missing_keywords)
        elif signal.quality == QualityLevel.STUCK:
            q = self.strategy.next_hint(topic)
        elif signal.quality == QualityLevel.CONFLICT:
            q = self._confirm_question(signal)  # TODO(담당 C)
        else:  # SUFFICIENT
            q = self.strategy.next_question(last_signal=signal)

        self._advance_to(q)
        return q

    def _on_stuck(self) -> Question | None:
        """음성 침묵(막힘) → 힌트성 질문."""
        q = self.strategy.next_hint(self.session.current_question.topic)
        self._advance_to(q)
        return q

    def _advance_to(self, question: Question) -> None:
        self.session.current_question = question
        self.session.asked_count += 1

    def _confirm_question(self, signal) -> Question:
        """이전 답변과 충돌 시 확인 질문. TODO(담당 C)."""
        raise NotImplementedError
