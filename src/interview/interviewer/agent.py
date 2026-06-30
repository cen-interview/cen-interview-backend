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
    InterviewerEvent,
    NoResponseTimeout,
    ReplayRequested,
    SilenceDetected,
)
from interview.schemas.question import Question
from interview.schemas.signals import AnswerQualitySignal
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

    def handle(self, event: InterviewerEvent) -> Question | None:
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
        # AnswerSubmitted 는 delivery_metrics 를 dict 로 묶어 갖고 있지 않고
        # speech_rate_wpm / filler_count 로 따로 갖고 있다 (채팅 모드면 둘 다 None).
        delivery_metrics = None
        if event.speech_rate_wpm is not None or event.filler_count is not None:
            delivery_metrics = {
                "speech_rate_wpm": event.speech_rate_wpm,
                "filler_count": event.filler_count,
            }
        signal = self.assessment.evaluate(
            question=self.session.current_question,
            answer_text=event.text,
            delivery_metrics=delivery_metrics,
        )
        if self.session.is_done():
            self.session.finished = True
            return None

        topic = self.session.current_question.topic
        if signal.quality == "shallow":
            q = self.strategy.next_follow_up(topic, signal.missing_keywords)
        elif signal.quality == "stuck":
            q = self.strategy.next_hint(topic)
        elif signal.quality == "conflict":
            q = self._confirm_question(signal)  # TODO(담당 C)
        else:  # sufficient
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

    def _confirm_question(self, signal: AnswerQualitySignal) -> Question:
        """이전 답변과 충돌 시 확인 질문."""
        # [현재 Stub 작동] 실제 LLM 문구 생성 대신 고정 템플릿 사용
        current = self.session.current_question
        return Question(
            question_id=f"q_confirm_{self.session.asked_count}",
            text=(
                f"[확인] 방금 답변이 이전 답변(질문 {signal.conflict_with_question_id})과 "
                "다소 다른 것 같습니다. 어느 쪽이 맞는지 다시 한번 설명해주실 수 있나요?"
            ),
            topic=current.topic,
            difficulty=current.difficulty,
            kind="confirm",
            parent_question_id=signal.conflict_with_question_id,
        )
