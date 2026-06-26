"""Assessment Agent.

답변을 근거 기반으로 평가하고 강점/약점을 누적하며, 종료 후 최종 평가서를 만든다.
매 답변마다 evaluate() 가 호출되고, 마지막에 finalize() 가 호출된다.
"""

from interview.schemas.question import Question
from interview.schemas.report import AnswerEvaluation, CompetencyModel, FinalReport
from interview.schemas.signals import AnswerQualitySignal, QualityLevel
from interview.assessment import evaluator, report_builder


class AssessmentAgent:
    def __init__(self) -> None:
        self.competency = CompetencyModel()
        self.evaluations: list[AnswerEvaluation] = []

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
        signal, score = evaluator.judge_answer(
            question, answer_text, delivery_metrics
        )
        self.competency.record(question.topic, score)
        self.evaluations.append(
            AnswerEvaluation(
                question_id=question.question_id,
                topic=question.topic,
                answer_text=answer_text,
                signal=signal,
                score=score,
            )
        )
        return signal

    def finalize(self) -> FinalReport:
        """면접 종료 후 최종 평가서 생성."""
        return report_builder.build_report(self.competency, self.evaluations)
