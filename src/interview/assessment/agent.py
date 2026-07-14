"""답변 평가 상태와 질문 세트 단위 채점을 관리하는 Assessment Agent.

이 모듈은 Interviewer로부터 질문과 사용자 답변을 전달받아 답변 하나를
평가하고, 메인 질문과 파생 질문을 하나의 질문 세트로 묶어 최종 점수를
계산한다. 면접이 종료되면 누적된 문항별 평가를 이용해 FinalReport를 생성한다.

주요 처리 흐름:
    1. evaluate():
        질문과 답변을 Assessment LangGraph에 전달한다.
        LangGraph가 반환한 AnswerQualitySignal을 AnswerAttempt로 저장한다.

    2. complete_question_set():
        메인 질문과 follow_up, challenge, confirm, trap 등의 파생 답변을
        하나의 질문 세트로 묶어 점수를 계산한다.
        계산 결과는 AnswerEvaluation과 CompetencyModel에 누적한다.

    3. finalize():
        누적된 문항별 평가와 역량 상태를 report_builder에 전달하여
        최종 면접 리포트를 생성한다.

상태 관리:
    current_attempts:
        현재 질문 세트에 속한 답변 시도 목록.
        질문 세트 평가가 완료되면 초기화된다.

    all_attempts:
        면접 전체의 답변 이력.
        질문 세트가 끝나도 유지되며 이전 답변과의 모순 검사에 사용된다.

    evaluations:
        완료된 질문 세트별 최종 평가 목록.

    competency:
        주제별 점수, 평균 점수, 강점과 보완 포인트를 관리하는 누적 역량 모델.

평가 단위:
    AnswerQualitySignal:
        답변 하나의 정확도, 충분성, quality와 다음 질문 방향을 나타낸다.

    AnswerAttempt:
        질문 하나와 답변 하나, 그리고 해당 답변의 평가 신호를 기록한다.

    AnswerEvaluation:
        메인 질문과 파생 질문을 합친 질문 세트의 최종 점수와 평가 내용이다.

    FinalReport:
        전체 면접의 종합 점수, 요약, 강점, 보완점, 학습 추천과
        문항별 평가를 포함한다.
"""

from interview.assessment import report_builder
from interview.assessment.scoring import AnswerAttempt, score_question_set
from interview.schemas.question import (
    Question, 
    QuestionKind,
    )
from interview.schemas.report import (
    AnswerEvaluation,
    CompetencyModel,
    FinalReport,
    QualityTrace,
)
from interview.schemas.signals import AnswerQualitySignal
from interview.assessment.graph import AssessmentState, get_compiled_graph

class AssessmentAgent:
    """답변 평가 상태를 관리하는 Assessment Agent.

    Attributes:
        competency:
            면접 전체의 누적 역량 상태.
            주제별 점수, 전체 강점, 전체 보완 포인트 등을 저장한다.

        evaluations:
            문항별 최종 평가 결과 목록.
            메인 질문 1개와 그에 연결된 파생 질문 답변을 묶은
            AnswerEvaluation이 누적된다.

        current_attempts:
            현재 질문 세트의 답변 시도 목록.
            메인 질문 답변과 follow_up, challenge, confirm, trap 등의
            파생 질문 답변을 임시로 저장한다.
            complete_question_set()이 호출되면 평가 생성 후 비워진다.

        all_attempts:
            면접 전체 답변 이력.
            질문 세트가 끝나도 삭제되지 않는다.
            이전 답변과 현재 답변의 모순 감지에 사용된다.
    """

# 답변 평가와 최종 리포트 생성에 필요한 내부 상태를 초기화한다.
    def __init__(self, user_id: str | None = None) -> None:
        self.user_id = user_id
        self.competency = CompetencyModel()
        self.evaluations: list[AnswerEvaluation] = []

        # 현재 질문 세트의 답변 시도
        # 메인 질문 + follow_up / challenge / confirm / trap 등을 묶어 점수 산정할 때 사용
        self.current_attempts: list[AnswerAttempt] = []

        # 면접 전체 답변 이력
        # 이전 답변과 현재 답변의 모순 여부를 확인할 때 사용
        self.all_attempts: list[AnswerAttempt] = []

# 답변 하나를 평가하고 다음 면접 흐름을 결정할 평가 신호를 반환한다.
    def evaluate(
        self,
        question: Question,
        answer_text: str,
        delivery_metrics: dict | None = None,
    ) -> AnswerQualitySignal:

        """사용자 답변 하나를 평가하고 평가 신호를 반환한다.

        Args:
            question:
                현재 사용자가 답변한 질문 객체.
                question_id, text, topic, kind, category, difficulty 값을 가진다.

            answer_text:
                사용자가 제출한 답변 텍스트.
                채팅이면 입력 문자열이고, 음성이면 STT 결과 텍스트이다.

            delivery_metrics:
                음성 면접에서 전달력 평가를 위해 사용하는 보조 데이터.
                예: speech_rate_wpm, filler_count.
                채팅 모드에서는 None일 수 있다.

        Returns:
            AnswerQualitySignal:
                Interviewer가 다음 면접 흐름을 결정할 때 사용하는 평가 신호.
                quality 값에 따라 다음 메인 질문, 꼬리 질문, 압박 질문,
                확인 질문, 함정 질문 등의 흐름이 결정된다.
        """
        state = AssessmentState(
            question=question,
            answer_text=answer_text,
            delivery_metrics=delivery_metrics,
            history=self.all_attempts,
            user_id=self.user_id,
        )

        result_state = get_compiled_graph().invoke(state)

        signal = result_state["final_signal"]

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

# 현재 메인 질문과 파생 질문을 묶어 하나의 문항 평가를 생성한다.
    def complete_question_set(
        self,
        main_question_id: str,

    ) -> None:
        """현재 질문 세트를 하나의 AnswerEvaluation으로 저장한다.

        질문 세트란 메인 질문 1개와 그 질문에서 파생된 follow_up,
        challenge, confirm, trap 질문 및 각 답변을 묶은 단위이다.

        Args:
            main_question_id:
                현재 질문 세트의 기준이 되는 메인 질문 ID.
                최종 문항 평가는 이 메인 질문 ID 기준으로 저장된다.

        처리 흐름:
            1. current_attempts에 저장된 답변 시도를 확인한다.
            2. score_question_set()으로 질문 세트 최종 점수를 계산한다.
            3. 메인 질문에 해당하는 AnswerAttempt를 찾는다.
            4. 메인 답변과 파생 질문 답변을 합쳐 answer_summary를 만든다.
            5. AnswerEvaluation을 생성해 evaluations에 저장한다.
            6. topic_scores에 주제별 점수를 반영한다.
            7. current_attempts를 비워 다음 질문 세트를 준비한다.
        """

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
            answer_summary=self._build_answer_context(),
            score=score.score,
            comment=score.comment,
            delivery_note=self._build_delivery_note(),
            quality_trace=self._build_quality_trace(),
        )

        self.evaluations.append(evaluation)
        self.competency.topic_scores[main_attempt.question_topic] = score.score
        self._update_competency_model()
        
        self.current_attempts.clear()

# 누적된 문항 평가 점수로 전체 평균 점수를 갱신한다.        
    def _update_competency_model(self) -> None:

        if not self.evaluations:
            self.competency.average_score = 0
            return

        self.competency.average_score = round(
            sum(
                evaluation.score
                for evaluation in self.evaluations
            )
            / len(self.evaluations),
            0,
        )

# 현재 질문 세트에서 기준이 되는 메인 질문 답변을 찾는다.
    def _find_main_attempt(
        self,
        main_question_id: str,
    ) -> AnswerAttempt:
        """현재 질문 세트에서 메인 질문 답변을 찾는다.

        Args:
            main_question_id:
                찾고자 하는 메인 질문 ID.

        Returns:
            AnswerAttempt:
                main_question_id와 QuestionKind.MAIN을 모두 만족하는 답변 시도.
                정상 흐름에서는 반드시 메인 질문 답변이 존재해야 한다.

        Note:
            임시 호환 처리를 위해 메인 질문을 찾지 못하면
            current_attempts의 첫 번째 답변을 반환한다.
        """


        for attempt in self.current_attempts:
            if (
                attempt.question_id == main_question_id
                and attempt.question_kind == QuestionKind.MAIN
            ):
                return attempt

        # 기존 데이터와의 임시 호환을 위해 첫 답변을 사용한다.
        return self.current_attempts[0]

# 현재 질문 세트의 quality 판정 이력을 생성한다.    
    def _build_quality_trace(self) -> list[QualityTrace]:
        return [
            QualityTrace(
                question_kind=attempt.question_kind.value,
                quality=attempt.signal.quality.value,
                target=attempt.signal.next_probe_target,
                rationale=attempt.signal.rationale,
            )
            for attempt in self.current_attempts
        ]

# 현재 질문 세트의 전달력 평가 문장을 중복 없이 합친다.    
    def _build_delivery_note(self) -> str | None:
        notes = [
            attempt.signal.delivery_note
            for attempt in self.current_attempts
            if attempt.signal.delivery_note
        ]

        if not notes:
            return None

        return " ".join(dict.fromkeys(notes))

# 현재 질문 세트에 포함된 답변 내용을 하나의 문자열로 합친다.
    def _build_answer_context(self) -> str:   
        return "\n".join(
            attempt.answer_text
            for attempt in self.current_attempts
    )

# 누적된 문항 평가를 바탕으로 최종 면접 리포트를 생성한다.
    def finalize(self) -> FinalReport:
        """면접 종료 후 최종 평가서를 생성한다.

        Returns:
            FinalReport:
                면접 전체 요약, 종합 점수, 전체 강점, 전체 보완 포인트,
                추천 학습 방향, 문항별 평가 목록을 포함한 최종 리포트.
        """

        return report_builder.build_report(
            self.competency,
            self.evaluations,
        )
