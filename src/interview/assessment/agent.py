"""Assessment Agent.

답변을 근거 기반으로 평가하고 강점/약점을 누적하며, 종료 후 최종 평가서를 만든다.
매 답변마다 evaluate() 가 호출되고, 마지막에 finalize() 가 호출된다.



evaluator.py
→ 답변 1개 품질 판단

scoring.py
→ 질문 세트 단위 점수 산정

agent.py
→ 답변 시도 누적 + 세트 종료 시 평가 저장
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
            question, answer_text, delivery_metrics
        )

        self.current_attempts.append(
            AnswerAttempt(
                question_id=question.question_id,
                answer_text=answer_text,
                signal=signal,
            )
        )

        return signal
    
    def complete_question_set(self, topic: str, main_question_id: str) -> None:
        """메인 질문 + 후속 질문 답변을 합쳐 최종 평가를 누적한다."""

        score = score_question_set(self.current_attempts)

        evaluation = AnswerEvaluation(
            question_id=main_question_id,
            topic=topic,
            quality=score.final_quality,
            accuracy=score.accuracy,
            sufficiency=score.sufficiency,
            key_concepts=[],
            comment=score.comment,
            delivery_note=None,
        )

        self.evaluations.append(evaluation)
        self.competency.topic_scores[topic] = score.accuracy

        self.current_attempts.clear()

    def finalize(self) -> FinalReport:
        """면접 종료 후 최종 평가서 생성."""
        return report_builder.build_report(self.competency, self.evaluations)
