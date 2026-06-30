"""Assessment Agent.

답변을 근거 기반으로 평가하고 강점/약점을 누적하며, 종료 후 최종 평가서를 만든다.
매 답변마다 evaluate() 가 호출되고, 마지막에 finalize() 가 호출된다.
"""

from interview.schemas.question import Question
from interview.schemas.report import AnswerEvaluation, CompetencyModel, FinalReport
from interview.schemas.signals import AnswerQualitySignal
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
          - evaluator.check_conflict 로 충돌 시 signal.quality=CONFLICT 보정
          - 강점/약점 산정 로직 고도화 (지금은 점수 임계값으로 단순 분류)
        """
        signal, score = evaluator.judge_answer(
            question, answer_text, delivery_metrics
        )

        # [현재 Stub 작동] CompetencyModel 은 메서드 없는 순수 데이터 모델이라
        # 여기서 직접 topic_scores 를 갱신하고 강점/약점을 재계산한다.
        self.competency.topic_scores[question.topic] = score
        self.competency.strengths = [
            t for t, s in self.competency.topic_scores.items() if s >= 0.7
        ]
        self.competency.weaknesses = [
            t for t, s in self.competency.topic_scores.items() if s < 0.5
        ]

        self.evaluations.append(
            AnswerEvaluation(
                question_id=question.question_id,
                topic=question.topic,
                quality=signal.quality,
                accuracy=score,
                sufficiency=score,
                key_concepts=signal.covered_keywords,
                comment=signal.rationale,
                delivery_note="[Stub] 전달력 보조 평가 미구현" if delivery_metrics else None,
            )
        )
        return signal

    def finalize(self) -> FinalReport:
        """면접 종료 후 최종 평가서 생성."""
        return report_builder.build_report(self.competency, self.evaluations)
