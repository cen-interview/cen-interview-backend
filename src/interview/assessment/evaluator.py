"""답변 채점 로직.

한 답변을 받아 근거를 조회하고, 답변 품질/분기 신호를 만든다.
점수는 여기서 계산하지 않고 scoring.py에서 질문 세트 단위로 계산한다.
"""

import random
from uuid import uuid4

from pydantic import BaseModel, Field

from interview.evidence.retrieval import search_evidence

from interview.schemas.evidence import EvidenceChunk
from interview.schemas.question import (
    Question,
    QuestionCategory,
    QuestionKind,
)

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

class JudgeResult(BaseModel):
    """LLM이 생성해야 하는 답변 평가 결과."""

    quality: AnswerQuality

    next_probe_target: str | None = None
    # quality 판정에 영향을 준 핵심 키워드
    rationale: list[str] = Field(default_factory=list)

def judge_answer(
    question: Question,
    answer_text: str,
    delivery_metrics: dict | None = None,
) -> AnswerQualitySignal:



    _ = search_evidence(
        query=question.text,
        topic=question.topic,

    )

    return AnswerQualitySignal(
        answer_id=f"answer-{uuid4()}",
        question_id=question.question_id,
        quality=judge_result.quality,
        next_probe_target=judge_result.next_probe_target,
        rationale=judge_result.rationale,
    )


def _judge_with_llm(
    question: Question,
    answer_text: str,
    evidence_chunks: list[EvidenceChunk],
    delivery_metrics: dict | None = None,
) -> JudgeResult:
    """질문, 답변, Evidence를 비교하여 답변을 평가한다."""

    # TODO: 실제 LLM 연결 시 아래 값들을 프롬프트로 전달
    _ = answer_text
    _ = evidence_chunks
    _ = delivery_metrics

    return _temporary_judge_result(question)

def _temporary_judge_result(
    question: Question,
) -> JudgeResult:
    """LLM 연결 전 랜덤한 임시 평가 결과를 반환한다."""

    _ = question

    temporary_results = [
        # index 0
        JudgeResult(
            quality=AnswerQuality.SUFFICIENT,
            next_probe_target=None,
            rationale=[
                "핵심 내용 설명 완료",
                "추가 확인 불필요",
            ],
        ),


        # index 1
        JudgeResult(
            quality=AnswerQuality.BONUS_AVAILABLE,
            next_probe_target="실제 프로젝트 적용 사례",
            rationale=[
                "기본 개념 설명 확인",
                "실제 적용 사례 부족",
            ],
        ),

        # index 2
        JudgeResult(
            quality=AnswerQuality.MISCONCEPTION,
            next_probe_target="핵심 개념의 정확한 역할",
            rationale=[
                "핵심 개념 오해",
                "역할 설명 오류",
            ],
        ),

        # index 3
        JudgeResult(
            quality=AnswerQuality.CONFIRM_POSITIVE,
            next_probe_target="기술의 적용 범위",
            rationale=[
                "설명은 대체로 정확함",
                "적용 범위 확인 필요",
            ],
        ),

        # index 4
        JudgeResult(
            quality=AnswerQuality.CONFIRM_NEGATIVE,
            next_probe_target="기존 설명과 충돌하는 부분",
            rationale=[
                "근거와 일부 불일치",
                "사실관계 재확인 필요",
            ],
        ),

        # index 5
        JudgeResult(
            quality=AnswerQuality.TRAP_AVAILABLE,
            next_probe_target="유사 개념의 차이",
            rationale=[
                "유사 개념 혼동 가능성",
                "개념 구분 확인 필요",
            ],
        ),
    ]

    return random.choice(temporary_results)

