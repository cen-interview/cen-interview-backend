"""Interviewer가 외부 에이전트에 요구하는 최소 계약.

Interviewer는 Strategy와 Assessment의 구체 구현을 직접 알지 않고 이 모듈의
Protocol에만 의존한다. Protocol은 명시적인 상속을 요구하지 않으므로 같은
메서드 시그니처를 제공하는 실제 에이전트와 테스트 fake를 모두 사용할 수 있다.
"""

from typing import Protocol

from interview.schemas.question import Question
from interview.schemas.report import FinalReport
from interview.schemas.rubric import RubricCandidate
from interview.schemas.signals import AnswerQualitySignal


class StrategyPort(Protocol):
    """Interviewer가 질문 생성을 위해 사용하는 Strategy 계약."""

    def next_question(
        self,
        last_signal: AnswerQualitySignal | None,
    ) -> Question:
        """직전 평가 신호를 반영한 다음 메인 질문을 반환한다.

        Args:
            last_signal:
                직전 답변의 평가 신호. 첫 질문이면 None.

        Returns:
            다음에 제시할 메인 질문.
        """
        ...

    def next_follow_up(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        """추가 설명을 확인할 꼬리 질문을 반환한다.

        Args:
            topic:
                현재 질문 세트의 주제.

            parent_question_id:
                파생 질문이 연결될 부모 질문 ID.

            target:
                다음 질문에서 집중적으로 확인할 대상.

            answer_excerpt:
                질문 생성에 참고할 직전 답변의 일부.

        Returns:
            부모 질문에 연결된 꼬리 질문.
        """
        ...

    def next_challenge(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        """오개념이나 논리적 허점을 확인할 압박 질문을 반환한다.

        Args:
            topic:
                현재 질문 세트의 주제.

            parent_question_id:
                파생 질문이 연결될 부모 질문 ID.

            target:
                다음 질문에서 집중적으로 확인할 대상.

            answer_excerpt:
                질문 생성에 참고할 직전 답변의 일부.

        Returns:
            부모 질문에 연결된 압박 질문.
        """
        ...

    def next_confirm_positive(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        """대체로 맞는 답변의 범위를 재확인할 질문을 반환한다.

        Args:
            topic:
                현재 질문 세트의 주제.

            parent_question_id:
                파생 질문이 연결될 부모 질문 ID.

            target:
                다음 질문에서 집중적으로 확인할 대상.

            answer_excerpt:
                질문 생성에 참고할 직전 답변의 일부.

        Returns:
            부모 질문에 연결된 긍정 확인 질문.
        """
        ...

    def next_confirm_negative(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        """근거나 이전 답변과 충돌하는 내용을 재확인할 질문을 반환한다.

        Args:
            topic:
                현재 질문 세트의 주제.

            parent_question_id:
                파생 질문이 연결될 부모 질문 ID.

            target:
                다음 질문에서 집중적으로 확인할 대상.

            answer_excerpt:
                질문 생성에 참고할 직전 답변의 일부.

        Returns:
            부모 질문에 연결된 부정 확인 질문.
        """
        ...

    def next_trap(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        """혼동하기 쉬운 개념을 구분하는지 확인할 질문을 반환한다.

        Args:
            topic:
                현재 질문 세트의 주제.

            parent_question_id:
                파생 질문이 연결될 부모 질문 ID.

            target:
                다음 질문에서 집중적으로 확인할 대상.

            answer_excerpt:
                질문 생성에 참고할 직전 답변의 일부.

        Returns:
            부모 질문에 연결된 함정 질문.
        """
        ...

    def next_hint(
        self,
        question: Question,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        """지원자가 답변을 이어가기 어려울 때 사용할 힌트 질문을 반환한다.

        Args:
            question:
                힌트를 제공할 현재 질문.

            target:
                힌트가 집중할 선택적 확인 대상.

            answer_excerpt:
                힌트 생성에 참고할 선택적 답변 일부. 완전한 침묵에서는 None.

        Returns:
            현재 질문에서 파생된 힌트 성격의 질문.
        """
        ...


class AssessmentPort(Protocol):
    """Interviewer가 답변 평가와 세션 마무리에 사용하는 Assessment 계약."""

    def evaluate(
        self,
        question: Question,
        answer_text: str,
        delivery_metrics: dict | None = None,
    ) -> AnswerQualitySignal:
        """현재 질문에 대한 답변을 평가하고 흐름 결정 신호를 반환한다.

        Args:
            question:
                지원자가 답변한 현재 질문.

            answer_text:
                지원자가 제출한 답변 본문.

            delivery_metrics:
                음성 답변의 발화 속도 등 선택적 전달 지표.

        Returns:
            다음 질문 흐름을 결정하는 답변 품질 신호.
        """
        ...

    def complete_question_set(
        self,
        main_question_id: str,
    ) -> RubricCandidate | None:
        """현재 메인 질문과 파생 질문 묶음의 평가를 완료한다.

        Args:
            main_question_id:
                완료할 질문 세트의 기준 메인 질문 ID.
        """
        ...

    def finalize(self) -> FinalReport:
        """누적된 평가를 바탕으로 최종 면접 리포트를 반환한다.

        Returns:
            면접 전체의 최종 평가 리포트.
        """
        ...
