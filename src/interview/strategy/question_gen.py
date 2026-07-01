"""질문 생성 로직 (LLM + 근거).

근거를 조회해 실제 질문 문장을 만든다. agent.py 는 이 함수들을 배선만 하고,
실제 "어떻게 생성하느냐"는 여기에 둔다.
"""
from uuid import uuid4

from interview.evidence.retrieval import search_evidence
from interview.llm import get_llm
from interview.schemas.question import Difficulty, Question,QuestionKind
from interview.strategy import prompts


def generate_question(topic: str, difficulty: Difficulty) -> Question:
    """주제 + 난이도로 일반 질문 생성.

    TODO(담당 B):
      - search_evidence(topic=...) 로 근거 chunk 조회
      - prompts.QUESTION_GEN_SYSTEM + 근거로 LLM 호출
      - linked_evidence 에 사용한 chunk_id 기록
    """
    evidence_chunks = search_evidence(query=topic, topic=topic)

    return Question(
        question_id=str(uuid4()),
        text=f"{topic}에 대해 설명해 주세요.",
        topic=topic,
        difficulty=difficulty,
        kind=QuestionKind.MAIN,
        evidence_ids=[chunk.chunk_id for chunk in evidence_chunks],
        parent_question_id=None,
    )



def generate_follow_up(topic: str, missing_keywords: list[str]) -> Question:
    """답변이 얕을 때 부족한 키워드를 끌어내는 꼬리 질문 생성.

    설계 문서 예시:
      missing=["fetch join","지연 로딩"] → "N+1 이 지연 로딩 상황에서 어떻게
      발생하는지 설명하고, fetch join 으로 어떻게 줄이는지 말해보세요."

    TODO(담당 B): prompts.FOLLOW_UP_SYSTEM + 근거 + missing_keywords 로 생성
    """
    keyword_text = ", ".join(missing_keywords) if missing_keywords else "핵심 개념"

    evidence_chunks = search_evidence(query=keyword_text, topic=topic)

    return Question(
        question_id=str(uuid4()),
        text=f"{topic} 답변에서 {keyword_text} 부분을 조금 더 설명해 주세요.",
        topic=topic,
        difficulty=Difficulty.EASY,
        kind=QuestionKind.FOLLOW_UP,
        evidence_ids=[chunk.chunk_id for chunk in evidence_chunks],
        parent_question_id=None,
    )



def generate_confirm(
    topic: str,
    misconception_note: str | None = None,
) -> Question:
    """오개념이 의심될 때 확인 질문 생성."""
    query = misconception_note or topic
    evidence_chunks = search_evidence(query=query, topic=topic)

    detail = (
        f" 특히 '{misconception_note}' 부분을 기준으로 다시 생각해 주세요."
        if misconception_note
        else ""
    )

    return Question(
        question_id=str(uuid4()),
        text=f"{topic} 답변에 오개념이 있을 수 있습니다.{detail} 다시 설명해 주시겠어요?",
        topic=topic,
        difficulty=Difficulty.EASY,
        kind=QuestionKind.CONFIRM,
        evidence_ids=[chunk.chunk_id for chunk in evidence_chunks],
        parent_question_id=None,
    )
