"""답변 채점 로직.

한 답변을 받아 근거를 조회하고, 답변 품질 신호를 만든다.
점수는 여기서 계산하지 않고 scoring.py에서 질문 세트 단위로 계산한다.
"""

from interview.evidence.retrieval import search_evidence
from interview.llm import get_llm
from interview.schemas.question import Question, QuestionKind
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal
from interview.assessment import prompts

"""
답변 부족 / 누락 있음
→ quality = "shallow"
→ Strategy.next_follow_up()

오개념 / 이전 답변과 충돌
→ quality = "conflict"
→ Strategy.next_confirm()

충분한 답변
→ quality = "sufficient"
→ Strategy.next_question()
→ 내부에서 _pick_topic()
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

    # 후속 질문 답변은 일단 충분하다고 처리하는 임시 스텁
    if question.kind in (QuestionKind.FOLLOW_UP, QuestionKind.CONFIRM):
        return AnswerQualitySignal(
            question_id=question.question_id,
            quality=AnswerQuality.SUFFICIENT,
            missing_keywords=[],
            covered_keywords=["핵심 개념"],
            misconception_note=None,
            rationale="임시 평가: 후속 답변을 통해 부족한 내용을 보완했습니다.",
        )

    # 메인 질문은 일단 얕다고 처리하는 임시 스텁
    return AnswerQualitySignal(
        question_id=question.question_id,
        quality=AnswerQuality.SHALLOW,
        missing_keywords=["핵심 개념"],
        covered_keywords=[],
        misconception_note=None,
        rationale="임시 평가: 답변은 일부 맞지만 핵심 개념 설명이 부족합니다.",
    )


# def check_conflict(
#     question: Question, answer_text: str, history: list
# ) -> str | None:
#     """이전 답변과 충돌하는지 확인. 충돌하면 충돌한 question_id 반환.

#     TODO(담당 D): prompts.CONFLICT_CHECK_SYSTEM 으로 이전 답변들과 대조
#     """
#     raise NotImplementedError
