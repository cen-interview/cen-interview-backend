"""답변 채점 로직.

Assessment에서 답변 하나를 평가하는 단계이다.

처리 흐름:
    1. 현재 질문과 관련된 Evidence를 조회한다.
    2. 질문, 답변, Evidence를 LLM에게 전달한다.
    3. LLM의 평가 결과를 JudgeResult로 생성한다.
    4. Interviewer가 사용할 AnswerQualitySignal을 반환한다.

주의:
    - 점수(score)는 계산하지 않는다.
    - 질문 세트(메인 질문 + 파생 질문)의 점수는 scoring.py에서 계산한다.
"""

import random
from uuid import uuid4

from pydantic import BaseModel, Field

from interview.evidence.retrieval import search_evidence

from interview.schemas.evidence import EvidenceChunk
from interview.schemas.question import (
    Question,

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
    """LLM이 생성하는 답변 평가 결과.

    Attributes:
        quality:
            답변 품질.
            Interviewer가 다음 질문 흐름을 결정하는 기준이다.

        next_probe_target:
            다음 질문에서 추가 확인할 대상.
            예)
                "Fetch Join"
                "지연 로딩"
                "트랜잭션 전파"

        rationale:
            quality를 판단한 근거.
            추후 평가 코멘트 생성에도 활용할 수 있다.
    """

    quality: AnswerQuality

    next_probe_target: str | None = None
    # quality 판정에 영향을 준 핵심 키워드
    rationale: list[str] = Field(default_factory=list)


# 평가
def judge_answer(
    question: Question,
    answer_text: str,
    delivery_metrics: dict | None = None,
    history: list | None = None,
) -> AnswerQualitySignal:
    
    """사용자 답변 하나를 평가하여 Interviewer용 평가 신호를 생성한다.

    Args:
        question:
            현재 사용자가 답변한 Question.

        answer_text:
            사용자의 답변.

        delivery_metrics:
            음성 전달력 평가를 위한 보조 정보.
            예)
                speech_rate_wpm
                filler_count

        history:
            지금까지의 전체 답변 이력.
            이전 답변과 현재 답변의 모순 여부를 판단할 때 사용한다.

    Returns:
        AnswerQualitySignal:
            Interviewer가 다음 질문 흐름을 결정하기 위한 평가 신호.
    """
    
    evidence_chunks = search_evidence(
    query=question.text,
    topic=question.topic,
)
    
    judge_result = _judge_with_llm(
    question=question,
    answer_text=answer_text,
    evidence_chunks=evidence_chunks,
    delivery_metrics=delivery_metrics,
    history=history,
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
    history: list | None = None,
) -> JudgeResult:
    """LLM을 이용하여 답변 하나를 평가한다.

    Args:
        question:
            현재 질문.

        answer_text:
            사용자 답변.

        evidence_chunks:
            Retrieval을 통해 검색된 근거 문서.

        delivery_metrics:
            음성 전달력 보조 정보.

        history:
            면접 전체 답변 이력.
            confirm_negative와 같은 모순 검출 시 활용한다.

    Returns:
        JudgeResult:
            답변 품질과 다음 질문 방향을 포함한 평가 결과.

    TODO:
        실제 구현 시 LLM에게 아래 정보를 전달한다.

        - Question
        - Answer
        - Evidence
        - Delivery Metrics
        - History
    """

    # TODO : 실제 LLM 호출
    _ = question
    _ = answer_text
    _ = evidence_chunks
    _ = delivery_metrics
    _ = history

    return _temporary_judge_result(question)

def _temporary_judge_result(
    question: Question,
) -> JudgeResult:
    """LLM 연결 전 임시 평가 결과를 생성한다.

    현재는 랜덤한 평가 결과를 반환하는 Stub이다.

    Args:
        question:
            현재 질문.

    Returns:
        JudgeResult:
            임시 평가 결과.
    """

    _ = question

    temporary_results = [
        
        # 답변이 충분한 경우
        JudgeResult(
            quality=AnswerQuality.SUFFICIENT,
            next_probe_target=None,
            rationale=[
                "핵심 내용 설명 완료",
                "추가 확인 불필요",
            ],
        ),


        # 꼬리 질문 가능
        JudgeResult(
            quality=AnswerQuality.BONUS_AVAILABLE,
            next_probe_target="실제 프로젝트 적용 사례",
            rationale=[
                "기본 개념 설명 확인",
                "실제 적용 사례 부족",
            ],
        ),

        # 오개념 존재
        JudgeResult(
            quality=AnswerQuality.MISCONCEPTION,
            next_probe_target="핵심 개념의 정확한 역할",
            rationale=[
                "핵심 개념 오해",
                "역할 설명 오류",
            ],
        ),

        # 긍정 확인 질문
        JudgeResult(
            quality=AnswerQuality.CONFIRM_POSITIVE,
            next_probe_target="기술의 적용 범위",
            rationale=[
                "설명은 대체로 정확함",
                "적용 범위 확인 필요",
            ],
        ),

        # 부정 확인 질문
        JudgeResult(
            quality=AnswerQuality.CONFIRM_NEGATIVE,
            next_probe_target="기존 설명과 충돌하는 부분",
            rationale=[
                "근거와 일부 불일치",
                "사실관계 재확인 필요",
            ],
        ),

        # 함정 질문 가능
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

