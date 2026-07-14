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

from uuid import uuid4

from pydantic import BaseModel, Field

from interview.evidence.retrieval import search_evidence

from interview.schemas.evidence import EvidenceChunk
from interview.schemas.question import (
    Question,
    QuestionCategory,
)
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal,ConflictType
from interview.llm.client import get_llm
from interview.llm.logging import log_llm_error, log_llm_output
from interview.assessment.prompts import JUDGE_SYSTEM,DELIVERY_NOTE

""" 분기 기준:


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

# LLM이 생성한 답변 품질, 정확도, 충분성과 후속 질문 정보를 담는다.
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

        accuracy:
            - 답변이 얼마나 정확한지 나타낸다.
            - 기술 개념이면 일반 지식 기준
            - 프로젝트 질문이면 Evidence와의 일치 기준
        
        sufficiency:
            - 질문에 필요한 범위를 얼마나 충분히 답했는지 나타낸다.
            - 맞는 말을 했더라도 설명 범위가 부족하면 낮아진다.
    """

    quality: AnswerQuality

    next_probe_target: str | None = None
    # quality 판정에 영향을 준 핵심 키워드
    rationale: list[str] = Field(default_factory=list)
    conflict_type: ConflictType | None = None

    # 이전 답변 또는 Evidence와 충돌이 의심되는지 여부
    # True면 정밀 충돌 검사 실행
    conflict_suspected: bool = False

    accuracy: float = Field(ge=0.0, le=1.0)
    sufficiency: float = Field(ge=0.0, le=1.0)

    delivery_note: str | None = Field(
    default=None,
    description=(
        "speech_rate_wpm과 filler_count만 해석한 전달력 평가 문장. "
        "기술 내용, 정답 여부, 정확성, 충분성, 해결 방법은 언급하지 않는다."
    ),
)


# 답변을 평가하고 Interviewer가 사용할 최종 평가 신호를 반환한다.
def judge_answer(
    question: Question,
    answer_text: str,
    delivery_metrics: dict | None = None,
    history: list | None = None,
    user_id: str | None = None,
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

        user_id:
            Evidence store에서 사용자별 namespace를 선택하기 위한 사용자 ID.

    Returns:
        AnswerQualitySignal:
            Interviewer가 다음 질문 흐름을 결정하기 위한 평가 신호.
    """
    
    evidence_chunks = _collect_evidence_for_question(
        question,
        answer_text,
        user_id=user_id,
    )
    
    judge_result = _judge_with_llm(
    question=question,
    answer_text=answer_text,
    evidence_chunks=evidence_chunks,
    delivery_metrics=delivery_metrics,
    history=history,
    
    )

    if judge_result.conflict_suspected:
        judge_result = _run_conflict_check(
            question=question,
            answer_text=answer_text,
            evidence_chunks=evidence_chunks,
            history=history,
            fallback_result=judge_result,
        )
    
    return AnswerQualitySignal(
        answer_id=f"answer-{uuid4()}",
        question_id=question.question_id,
        quality=judge_result.quality,
        next_probe_target=judge_result.next_probe_target,
        rationale=judge_result.rationale,
        accuracy=judge_result.accuracy,
        sufficiency=judge_result.sufficiency,
        conflict_type=judge_result.conflict_type,
        delivery_note=judge_result.delivery_note,
    )


# 질문, 답변, Evidence와 이전 이력을 LLM에 전달해 구조화된 평가를 생성한다.
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

    """

    
    history_summary = _build_history_summary(history)
    evidence_context = _build_evidence_context(question,evidence_chunks)
    
    delivery_context = _build_delivery_context(delivery_metrics)

    system_prompt = JUDGE_SYSTEM
    if delivery_context:
        system_prompt += f"\n\n{DELIVERY_NOTE}"


    output_fields = (
    "quality, next_probe_target, rationale, "
    "accuracy, sufficiency"
    )

    if delivery_context:
        output_fields += ", delivery_note"

    user_prompt = f"""

    [평가 대상 질문]
    question_id: {question.question_id}
    topic: {question.topic}
    category: {question.category.value}
    kind: {question.kind.value}
    difficulty: {question.difficulty.value}

    질문: {question.text}

    질문 종류: {question.kind.value}


    사용자 답변: {answer_text}

    {delivery_context}

    Evidence: {evidence_context}

    이전 답변 이력:
    {history_summary or "(없음)"}

    [평가 요청]
    위 답변을 category와 kind에 맞는 기준으로 평가하라.
    반드시 지정된 구조로 {output_fields}를 반환하라.
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(JudgeResult)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    
        if not delivery_context:
            result = result.model_copy(
            update={"delivery_note": None}
        )

        log_llm_output(
            "ANSWER_ASSESSMENT",
            result,
            metadata={
                "question_id": question.question_id,
                "topic": question.topic,
                "question_kind": question.kind.value,
                "category": question.category.value if question.category else None,
                "evidence_ids": [chunk.chunk_id for chunk in evidence_chunks],
            },
            input_data={
                "question": question,
                "answer_text": answer_text,
                "delivery_metrics": delivery_metrics,
                "history": history,
                "user_prompt": user_prompt,
            },
        )

        return result
    
    except Exception as e:
        log_llm_error(
            "ANSWER_ASSESSMENT",
            e,
            metadata={
                "question_id": question.question_id,
                "topic": question.topic,
                "question_kind": question.kind.value,
                "category": question.category.value if question.category else None,
                "evidence_ids": [chunk.chunk_id for chunk in evidence_chunks],
            },
            input_data={
                "question": question,
                "answer_text": answer_text,
                "delivery_metrics": delivery_metrics,
                "history": history,
                "user_prompt": user_prompt,
            },
        )
        raise RuntimeError("답변 평가 중 LLM 호출에 실패했습니다.") from e

# 검색된 Evidence를 LLM 평가 프롬프트에 사용할 문자열로 변환한다.
def _build_evidence_context(
    question: Question,
    evidence_chunks: list[EvidenceChunk],
) -> str:
    if evidence_chunks:
        return "\n".join(
            (
                f"- source_type: {chunk.source_type}\n"
                f"  source_url: {chunk.source_url}\n"
                f"  topic: {chunk.topic}\n"
                f"  confidence: {chunk.confidence}\n"
                f"  text: {chunk.text}"
            )
            for chunk in evidence_chunks
        )

    if question.category == QuestionCategory.PROJECT:
        return (
            "Evidence 없음 — 프로젝트 사실을 확인할 근거가 부족하다. "
            "Evidence에 없는 프로젝트 구현 사실은 단정하지 않는다."
        )

    return (
        "Evidence 없음 — project 질문은 프로젝트 구현 사실을 확인할 근거가 부족하다. "
        "Evidence에 없는 프로젝트 사실은 단정하지 말고, 확인 필요 또는 설명 부족으로 판단한다."
)

# 이전 답변 이력을 충돌 검사에 사용할 요약 문자열로 변환한다.
def _build_history_summary(history: list | None) -> str:
    if not history:
        return ""

    return "\n".join(
        f"- {attempt.question_topic} / {attempt.question_kind}: {attempt.answer_text}"
        for attempt in history
    )

# 현재 답변과 Evidence·이전 답변 사이의 충돌 여부를 정밀 검사한다.
def _run_conflict_check(
    question: Question,
    answer_text: str,
    evidence_chunks: list[EvidenceChunk],
    history: list | None = None,
    fallback_result: JudgeResult | None = None,
) -> JudgeResult:
    """이전 답변 및 Evidence와 현재 답변의 충돌 여부를 정밀 검사한다.

    Args:
        question:
            현재 질문.

        answer_text:
            현재 사용자 답변.

        evidence_chunks:
            현재 질문과 관련된 Evidence 목록.

        history:
            면접 전체 답변 이력.
            AssessmentAgent의 all_attempts가 전달된다.

    Returns:
        JudgeResult:
            충돌이 있으면 CONFIRM_NEGATIVE,
            충돌이 없으면 기존 judge 결과를 유지할 수 있도록 충분 또는 추가확인 결과를 반환한다.

    """

    _ = question
    _ = evidence_chunks

    if not history:
        if fallback_result is not None:
            return fallback_result.model_copy(
                update={"conflict_suspected": False}
            )

        return JudgeResult(
            quality=AnswerQuality.SUFFICIENT,
            next_probe_target=None,
            rationale=["이전 답변 이력이 없어 충돌 검사를 생략했습니다."],
            conflict_suspected=False,
            accuracy=0.85, sufficiency=0.8,
        )

    previous_answers = [
        attempt.answer_text
        for attempt in history
    ]

    # 임시 충돌 검사 규칙
    # 예: 이전 답변과 현재 답변에 서로 반대되는 표현이 있는지 확인
    conflict_pairs = [
        ("세션", "토큰"),
        ("동기", "비동기"),
        ("GET", "POST"),
        ("상태 유지", "무상태"),
        ("서버 저장", "클라이언트 저장"),
    ]

    for previous_answer in previous_answers:
        for left, right in conflict_pairs:
            previous_has_left = left in previous_answer
            previous_has_right = right in previous_answer
            current_has_left = left in answer_text
            current_has_right = right in answer_text

            if previous_has_left and current_has_right:
                return JudgeResult(
                    quality=AnswerQuality.CONFIRM_NEGATIVE,
                    next_probe_target=f"{left}와 {right}의 관계",
                    rationale=[
                        f"이전 답변에서는 '{left}'에 가깝게 설명했지만 현재 답변에서는 '{right}'에 가깝게 설명했습니다.",
                        "두 답변의 관계를 확인할 필요가 있습니다.",
                    ],
                    conflict_suspected=True,
                    accuracy=0.4, sufficiency=0.5,
                )

            if previous_has_right and current_has_left:
                return JudgeResult(
                    quality=AnswerQuality.CONFIRM_NEGATIVE,
                    next_probe_target=f"{left}와 {right}의 관계",
                    rationale=[
                        f"이전 답변에서는 '{right}'에 가깝게 설명했지만 현재 답변에서는 '{left}'에 가깝게 설명했습니다.",
                        "두 답변의 관계를 확인할 필요가 있습니다.",
                    ],
                    conflict_suspected=True,
                    accuracy=0.4, sufficiency=0.5,
                )

    if fallback_result is not None:
        return fallback_result.model_copy(
            update={"conflict_suspected": False}
        )

    return JudgeResult(
        quality=AnswerQuality.SUFFICIENT,
        next_probe_target=None,
        rationale=["이전 답변과 명확히 충돌하는 내용은 발견되지 않았습니다."],
        conflict_suspected=False,
        accuracy=0.85, sufficiency=0.8,
    )


# 프로젝트 질문인 경우에만 관련 Evidence를 검색해 반환한다.
def _collect_evidence_for_question(
    question: Question,
    answer_text: str,
    user_id: str | None = None,
) -> list[EvidenceChunk]:
    if question.category != QuestionCategory.PROJECT:
        return []

    return search_evidence(
        query=f"{question.text}\n{answer_text}",
        topic=question.topic,
        user_id=user_id,
    )

# 음성 전달력 지표를 delivery_note 생성용 프롬프트 문자열로 변환한다.
def _build_delivery_context(
    delivery_metrics: dict | None,
) -> str:
    if not delivery_metrics:
        return ""

    lines = ["[전달력 지표]"]

    speech_rate = delivery_metrics.get("speech_rate_wpm")
    filler_count = delivery_metrics.get("filler_count")

    if speech_rate is not None:
        lines.append(f"speech_rate_wpm: {speech_rate}")

    if filler_count is not None:
        lines.append(f"filler_count: {filler_count}")

    if len(lines) == 1:
        return ""

    return "\n".join(lines)
