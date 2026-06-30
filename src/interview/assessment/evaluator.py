"""답변 채점 로직 (근거 비교 + LLM-as-a-Judge).

한 답변을 받아 근거를 조회하고, LLM 으로 채점해 신호 + 점수를 만든다.
"""

from interview.evidence.retrieval import search_evidence
from interview.llm import get_llm
from interview.schemas.question import Question
from interview.schemas.signals import AnswerQualitySignal
from interview.assessment import prompts


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
    _ = search_evidence(query=question.text, topic=question.topic)
    _ = get_llm(temperature=0.0)

    # [현재 Stub 작동] LLM 채점 대신 답변 길이로 임시 판정
    length = len((answer_text or "").strip())
    if length == 0:
        quality, score = "stuck", 0.0
    elif length < 20:
        quality, score = "shallow", 0.4
    else:
        quality, score = "sufficient", 0.8

    signal = AnswerQualitySignal(
        question_id=question.question_id,
        quality=quality,
        missing_keywords=[] if quality == "sufficient" else ["핵심 개념"],
        covered_keywords=[] if quality != "sufficient" else ["핵심 개념"],
        rationale="[Stub] 답변 길이 기반 임시 판정 (실전엔 LLM judge로 교체)",
    )
    return signal, score


def check_conflict(
    question: Question, answer_text: str, history: list
) -> str | None:
    """이전 답변과 충돌하는지 확인. 충돌하면 충돌한 question_id 반환.

    TODO(담당 D): prompts.CONFLICT_CHECK_SYSTEM 으로 이전 답변들과 대조
    """
    # [현재 Stub 작동] 항상 충돌 없음으로 처리
    return None
