"""Interviewer Agent.

공통 이벤트 + Assessment 평가 신호 + 세션 상태를 보고 지금 면접을 어떻게
진행할지 결정한다. 답변 품질을 직접 평가하지 않는다. 그건 Assessment 역할이다.

흐름:
  answer_submitted → Assessment 평가 → signal.quality 로 라우팅
    SUFFICIENT     → 다음 질문   (Strategy.next_question)
    SHALLOW        → 꼬리 질문   (Strategy.next_follow_up)
    MISCONCEPTION  → 확인 질문   (Strategy.next_confirm)

  replay_requested     → 현재 질문 다시 제시
  end_requested        → 종료
  no_response_timeout  → 종료
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
from interview.schemas.question import Question, QuestionKind
from interview.schemas.signals import AnswerQuality
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

        TODO(담당 C): 아래 분기를 채운다. 핵심은 '한 벌의 흐름 로직'을 유지하는 것.
        """
        if isinstance(event, AnswerSubmitted):
            return self._on_answer(event)

        if isinstance(event, SilenceDetected):
            return self._on_silence(event)

        if isinstance(event, ReplayRequested):
            return self.session.current_question

        if isinstance(event, EndRequested):
            self.session.finished = True
            return None

        if isinstance(event, NoResponseTimeout):
            self.session.finished = True
            return None

        raise ValueError(f"unhandled event: {event}")

    def _on_answer(self, event: AnswerSubmitted) -> Question | None:
        """답변 처리: Assessment 에 평가 위임 후 신호로 라우팅."""

        delivery_metrics = {
            "speech_rate_wpm": event.speech_rate_wpm,
            "filler_count": event.filler_count,
        }
        
        current_question = self.session.current_question

        if current_question is None:
            self.session.finished = True
            return None
    
        signal = self.assessment.evaluate(
            question=current_question,
            answer_text=event.text,
            delivery_metrics=delivery_metrics,
        )

        topic = current_question.topic

        if signal.quality == AnswerQuality.SHALLOW:
            q = self.strategy.next_follow_up(
                topic=topic,
                missing_keywords=signal.missing_keywords,
            )

        elif signal.quality == AnswerQuality.MISCONCEPTION:
            q = self.strategy.next_confirm(
                topic=topic,
                misconception_note=signal.misconception_note,
            )

        else:  # AnswerQuality.SUFFICIENT
            self.assessment.complete_question_set(
                topic=self.session.main_topic or topic,
                main_question_id=self.session.main_question_id or current_question.question_id,
            )   

            if self.session.is_done():
                self.session.finished = True
                return None

            q = self.strategy.next_question(last_signal=signal)

            # 새 메인 질문 세트 시작
            self.session.main_question_id = q.question_id
            self.session.main_topic = q.topic

        self._advance_to(q)
        return q

    def _on_silence(self, event: SilenceDetected) -> Question | None:
        """음성 침묵 이벤트.

        hint 흐름이 제거되었으므로 현재 질문을 다시 제시한다.
        """
        return self.session.current_question

    def _advance_to(self, question: Question) -> None:
        self.session.current_question = question

        # 메인 질문만 목표 질문 수에 포함한다.
        if question.kind == QuestionKind.MAIN:
            self.session.asked_count += 1