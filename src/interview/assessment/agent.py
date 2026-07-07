"""Assessment Agent.

AssessmentAgent는 답변 평가 흐름의 중심 역할을 한다.

전체 역할:
  1. 사용자의 답변 1개를 evaluator.py에 넘겨 평가 신호를 받는다.
  2. 메인 질문과 파생 질문의 답변들을 current_attempts에 누적한다.
  3. 질문 세트가 끝나면 current_attempts를 하나의 AnswerEvaluation으로 저장한다.
  4. 면접 전체 답변 이력은 all_attempts에 계속 누적한다.
  5. 면접 종료 시 FinalReport를 생성한다.

질문 세트란:
  메인 질문 1개
  + follow_up / challenge / confirm / trap 등 파생 질문들
  + 각 질문에 대한 사용자 답변들

역할 분리:
  evaluator.py
    → 답변 하나의 quality와 rationale 판단

  scoring.py
    → 질문 세트 단위 점수 산정

  report_builder.py
    → 최종 리포트 생성

  agent.py
    → 답변 시도 누적 및 질문 세트 평가 저장



"""
      
        

from interview.assessment import evaluator, report_builder
from interview.assessment.scoring import AnswerAttempt, score_question_set
from interview.schemas.question import (
    Question, 
    QuestionKind,
    )
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

        # 현재 질문 세트의 답변 시도
        # 메인 질문 + follow_up / challenge / confirm / trap 등을 묶어 점수 산정할 때 사용
        self.current_attempts: list[AnswerAttempt] = []

        # 면접 전체 답변 이력
        # 이전 답변과 현재 답변의 모순 여부를 확인할 때 사용
        self.all_attempts: list[AnswerAttempt] = []

    def evaluate(
        self,
        question: Question,
        answer_text: str,
        delivery_metrics: dict | None = None,
    ) -> AnswerQualitySignal:

        # evaluator는 답변 하나를 보고 quality를 판단한다.
        # history=self.all_attempts를 넘기면 이전 답변과의 모순 여부도 평가에 활용할 수 있다.
        signal = evaluator.judge_answer(
            question=question,
            answer_text=answer_text,
            delivery_metrics=delivery_metrics,
            history=self.all_attempts,  # 나중에 evaluator에서 받도록 확장 가능
        )

        # AnswerAttempt는 "질문 1개에 대한 답변 시도 1건"을 기록하는 객체다.
        # 이후 질문 세트 단위 점수 산정(score_question_set)에 사용된다.
        attempt = AnswerAttempt(
            answer_id=signal.answer_id,
            question_id=question.question_id,
            # 질문 원문
            question_text=question.text,
            # 질문 주제
            question_topic=question.topic,
            # 질문 종류
            question_kind=question.kind,
            # 질문 카테고리
            question_category=question.category,
            # 질문 난이도
            question_difficulty=question.difficulty,
            # 답변 원문
            answer_text=answer_text,
            # 답변 평가
            signal=signal,
            # 음성 전달
            delivery_metrics=delivery_metrics,
        )

        self.current_attempts.append(attempt)
        self.all_attempts.append(attempt)

        return signal

    def complete_question_set(
        self,
        main_question_id: str,
    ) -> None:
        """메인 질문과 파생 질문 답변을 하나의 평가로 저장한다."""

        if not self.current_attempts:
            return

        score = score_question_set(self.current_attempts)

        main_attempt = self._find_main_attempt(main_question_id)

  
        evaluation = AnswerEvaluation(
            question_id=main_question_id,
             # 평가 주제
            topic=main_attempt.question_topic,
            # 메인 질문 원문.
            question=main_attempt.question_text,
            # 메인 질문에 대한 최초 답변.
            answer_summary=self._build_answer_summary(),
            score=score.score,
            comment=score.comment,
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
    
    def _build_answer_summary(self) -> str:
        """현재 질문 세트의 답변들을 하나의 답변으로 합친다.

        TODO:
      - 현재는 단순 연결
      - 추후 LLM으로 메인 답변 + 파생 답변을 하나의 자연스러운 답변으로 요약
        """
        return "\n".join(
            attempt.answer_text
            for attempt in self.current_attempts
    )

    def finalize(self) -> FinalReport:
        """면접 종료 후 최종 평가서를 생성한다."""

        return report_builder.build_report(
            self.competency,
            self.evaluations,
        )