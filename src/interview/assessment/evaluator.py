"""답변 채점 로직.

한 답변을 받아 근거를 조회하고, 답변 품질/분기 신호를 만든다.
점수는 여기서 계산하지 않고 scoring.py에서 질문 세트 단위로 계산한다.
"""

from interview.evidence.retrieval import search_evidence
from interview.schemas.question import Question, QuestionKind
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal

"""
분기 기준:

sufficient
→ 답변이 충분함
→ Strategy.next_question()

bonus_available
→ 답변은 맞지만 더 깊게 물어볼 요소가 있음
→ Strategy.next_follow_up()

misconception
→ 오개념 또는 논리적 허점이 있음
→ Strategy.next_challenge()

confirm_positive
→ 답변은 대체로 맞지만 범위/사실관계 확인이 필요함
→ Strategy.next_confirm_positive()

confirm_negative
→ Evidence 또는 이전 답변과 충돌함
→ Strategy.next_confirm_negative()

trap_available
→ 헷갈리기 쉬운 개념 구분 확인이 필요함
→ Strategy.next_trap()
"""



def judge_answer(
    question: Question,
    answer_text: str,
    delivery_metrics: dict | None = None,
) -> AnswerQualitySignal:
    """답변 1건을 평가해 Interviewer 라우팅용 신호를 반환한다."""

    _ = search_evidence(
        query=question.text,
        topic=question.topic,
    )

    # 후속 질문/압박/확인/함정 질문에 대한 답변은
    # 현재 스텁 단계에서는 질문 세트가 마무리된 것으로 처리한다.
    if question.kind in (
        QuestionKind.FOLLOW_UP,
        QuestionKind.CHALLENGE,
        QuestionKind.CONFIRM_POSITIVE,
        QuestionKind.CONFIRM_NEGATIVE,
        QuestionKind.TRAP,
    ):
        return AnswerQualitySignal(
            question_id=question.question_id,
            quality=AnswerQuality.SUFFICIENT,
            next_probe_target=None,
            rationale="임시 평가: 후속 질문 답변을 통해 현재 질문 세트를 마무리할 수 있다고 판단했습니다.",
        )

    # 메인 질문은 현재 스텁 단계에서 항상 꼬리 질문이 가능한 상태로 처리한다.
    return AnswerQualitySignal(
        question_id=question.question_id,
        quality=AnswerQuality.BONUS_AVAILABLE,
        next_probe_target="핵심 개념의 원인과 실제 적용 방식",
        rationale="임시 평가: 기본 답변은 가능하지만 원인, 사례, 한계점 등 추가 확인할 요소가 남아 있습니다.",
    )


