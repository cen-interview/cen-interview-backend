"""Assessment Agent.

답변 하나의 평가 결과를 저장하고,
메인 질문과 파생 질문을 질문 세트 단위로 묶어 최종 평가한다.
면접 종료 시 FinalReport를 생성한다.

역할 분리:
  evaluator.py
    → 답변 하나의 quality와 rationale 판단

  scoring.py
    → 질문 세트 단위 점수 산정

  agent.py
    → 답변 시도 누적 및 질문 세트 평가 저장
"""

from interview.assessment import evaluator, report_builder
from interview.assessment.scoring import AnswerAttempt, score_question_set
from interview.schemas.question import Question, QuestionKind
from interview.schemas.report import (
    AnswerEvaluation,
    CompetencyModel,
    FinalReport,
)
from interview.schemas.signals import AnswerQualitySignal


class AssessmentAgent:
    def __init__(self) -> None:
        self.competency = CompetencyModel()
        self.evaluations: list[AnswerEvaluation] = []
        self.current_attempts: list[AnswerAttempt] = []

    def evaluate(
        self,
        question: Question,
        answer_text: str,
        delivery_metrics: dict | None = None,
    ) -> AnswerQualitySignal:
        """답변 하나를 평가하고 결과를 누적한다.


        TODO(담당 D):
          - evaluator.check_conflict 로 충돌 시 signal.quality=CONFLICT 보정
          - 강점/약점 산정 로직 고도화 (지금은 점수 임계값으로 단순 분류)
        
        """

        signal = evaluator.judge_answer(
            question=question,
            answer_text=answer_text,
            delivery_metrics=delivery_metrics,
        )

        attempt = AnswerAttempt(
            answer_id=signal.answer_id,
            question_id=question.question_id,
            question_text=question.text,
            question_kind=question.kind,
            answer_text=answer_text,
            signal=signal,
            delivery_metrics=delivery_metrics,
        )

        self.current_attempts.append(attempt)

        return signal

    def complete_question_set(
        self,
        topic: str,
        main_question_id: str,
    ) -> None:
        """메인 질문과 파생 질문 답변을 하나의 평가로 저장한다."""

        if not self.current_attempts:
            return

        score = score_question_set(self.current_attempts)

        main_attempt = self._find_main_attempt(main_question_id)

        answer_ids = [
            attempt.answer_id
            for attempt in self.current_attempts
        ]

        derived_question_ids = [
            attempt.question_id
            for attempt in self.current_attempts
            if attempt.question_kind != QuestionKind.MAIN
        ]

        evaluation = AnswerEvaluation(
            question_id=main_question_id,
            topic=topic,
            question=main_attempt.question_text,
            answer=main_attempt.answer_text,
            answer_ids=answer_ids,
            derived_question_ids=derived_question_ids,
            quality=score.final_quality,
            score=score.score,
            accuracy=score.accuracy,
            sufficiency=score.sufficiency,
            strengths=score.strengths,
            improvements=score.improvements,
            comment=score.comment,
            delivery_note=None,
        )

        self.evaluations.append(evaluation)
        self.competency.topic_scores[topic] = score.score

        self.current_attempts.clear()

    def _find_main_attempt(
        self,
        main_question_id: str,
    ) -> AnswerAttempt:
        """현재 질문 세트에서 메인 질문의 답변을 찾는다."""

        for attempt in self.current_attempts:
            if (
                attempt.question_id == main_question_id
                and attempt.question_kind == QuestionKind.MAIN
            ):
                return attempt

        # 기존 데이터와의 임시 호환을 위해 첫 답변을 사용한다.
        return self.current_attempts[0]

    def finalize(self) -> FinalReport:
        """면접 종료 후 최종 평가서를 생성한다."""

        return report_builder.build_report(
            self.competency,
            self.evaluations,
        )