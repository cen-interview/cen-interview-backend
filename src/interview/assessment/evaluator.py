"""답변 채점 로직 (근거 비교 + LLM-as-a-Judge).

한 답변을 받아 근거를 조회하고, LLM 으로 채점해 신호 + 점수를 만든다.
"""

from interview.evidence.retrieval import search_evidence
from interview.llm import get_llm
from interview.schemas.question import Question
from interview.schemas.signals import AnswerQualitySignal
from interview.assessment import prompts

MIN_SUFFICIENT_WORDS = 8
MIN_SHALLOW_WORDS = 3


def judge_answer(
    question: Question,
    answer_text: str,
    delivery_metrics: dict | None = None,
) -> tuple[AnswerQualitySignal, float]:
    """답변을 채점해 (신호, 점수) 를 반환한다.

    TODO(담당 D):
      - search_evidence(query=question.text, topic=question.topic) 로 근거 조회
      - prompts.JUDGE_SYSTEM + 근거 + 답변으로 LLM 호출 (temperature 낮게)
      - 구조화 출력(JSON)을 파싱해 quality / missing_keywords / score 채우기
      - delivery_metrics 있으면 prompts.DELIVERY_NOTE 로 보조 반영 (음성)
    """
    chunks = search_evidence(query=question.text, topic=question.topic)

    # MVP fake: 실제 LLM 호출은 이후 단계에서 연결한다.
    # llm = get_llm(temperature=0.0)
    # prompt = prompts.JUDGE_SYSTEM + ...
    _ = get_llm
    _ = prompts.JUDGE_SYSTEM

    normalized = answer_text.strip()
    words = normalized.split()

    evidence_keywords = _evidence_keywords(chunks)
    covered_keywords = [
        keyword for keyword in evidence_keywords if keyword.lower() in normalized.lower()
    ]
    missing_keywords = [
        keyword for keyword in evidence_keywords if keyword not in covered_keywords
    ][:3]

    if not normalized or len(words) < MIN_SHALLOW_WORDS:
        quality = "stuck"
        score = 0.2
        rationale = "답변이 거의 없어 평가할 수 없습니다."
    elif len(words) < MIN_SUFFICIENT_WORDS:
        quality = "shallow"
        score = 0.45
        rationale = "답변은 있으나 설명이 짧아 추가 확인이 필요합니다."
    else:
        keyword_bonus = min(len(covered_keywords) * 0.08, 0.2)
        score = min(0.7 + keyword_bonus, 0.9)
        quality = "sufficient"
        rationale = "핵심 내용을 일정 수준 이상 설명했습니다."

    if delivery_metrics:
        rationale = f"{rationale} 전달 지표가 함께 수집되었습니다."

    return (
        AnswerQualitySignal(
            question_id=question.question_id,
            quality=quality,
            missing_keywords=missing_keywords,
            covered_keywords=covered_keywords,
            rationale=rationale,
        ),
        score,
    )


def check_conflict(
    question: Question, answer_text: str, history: list
) -> str | None:
    """이전 답변과 충돌하는지 확인. 충돌하면 충돌한 question_id 반환.

    TODO(담당 D): prompts.CONFLICT_CHECK_SYSTEM 으로 이전 답변들과 대조
    """
    _ = question
    _ = answer_text
    _ = history
    _ = prompts.CONFLICT_CHECK_SYSTEM
    return None


def _evidence_keywords(chunks: list) -> list[str]:
    keywords: list[str] = []
    for chunk in chunks:
        for token in chunk.topic.replace("/", " ").split():
            cleaned = token.strip()
            if cleaned and cleaned not in keywords:
                keywords.append(cleaned)
    return keywords[:5]
