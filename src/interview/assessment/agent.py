"""Assessment Agent.

답변을 근거 기반으로 평가하고, 질문 세트 단위로 평가 결과를 누적한다.
면접 종료 시 finalize()를 호출해 최종 평가서를 만든다.

역할 분리:
  evaluator.py
    → 답변 1개에 대한 품질/분기 신호 판단

  scoring.py
    → 메인 질문 + 후속 질문을 합친 질문 세트 단위 점수 산정

  agent.py
    → 답변 시도 누적 + 질문 세트 종료 시 평가 저장
"""

from interview.schemas.question import Question
from interview.schemas.report import AnswerEvaluation, CompetencyModel, FinalReport
from interview.schemas.signals import AnswerQualitySignal
from interview.assessment import evaluator, report_builder
from interview.assessment.scoring import AnswerAttempt, score_question_set

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
        """답변 1건 평가 → Interviewer 가 라우팅에 쓸 신호 반환.

        TODO(담당 D):
          - evaluator.judge_answer 로 (signal, score) 획득
          - evaluator.check_conflict 로 충돌 시 signal.quality=CONFLICT 보정
          - competency.record(topic, score), evaluations 누적
        """
        signal = evaluator.judge_answer(
            question=question,
            answer_text=answer_text,
            delivery_metrics=delivery_metrics,
        )

        self.current_attempts.append(
            AnswerAttempt(
                question_id=question.question_id,
                question_text=question.text,
                answer_text=answer_text,
                signal=signal,
            )
        )

        return signal
    
    def complete_question_set(
        self,
        topic: str,
        main_question_id: str,
    ) -> None:
        """메인 질문 + 후속 질문 답변을 합쳐 최종 평가를 누적한다."""

        if not self.current_attempts:
            return

        score = score_question_set(self.current_attempts)

        main_attempt = self.current_attempts[0]
        follow_up_question_ids = [
            attempt.question_id
            for attempt in self.current_attempts[1:]
        ]

        evaluation = AnswerEvaluation(
            question_id=main_question_id,
            topic=topic,
            question=main_attempt.question_text,
            answer=main_attempt.answer_text,
            quality=score.final_quality,
            score=score.score,
            accuracy=score.accuracy,
            sufficiency=score.sufficiency,
            follow_up_question_ids=follow_up_question_ids,
            strengths=score.strengths,
            improvements=score.improvements,
            comment=score.comment,
            delivery_note=None,
        )

        self.evaluations.append(evaluation)
        self.competency.topic_scores[topic] = score.score

        self.current_attempts.clear()

    def finalize(self) -> FinalReport:
        """면접 종료 후 최종 평가서 생성."""
        return report_builder.build_report(self.competency, self.evaluations)
