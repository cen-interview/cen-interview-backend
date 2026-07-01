"""질문 생성 로직 (LLM + 근거).

근거를 조회해 실제 질문 문장을 만든다. agent.py 는 이 함수들을 배선만 하고,
실제 "어떻게 생성하느냐"는 여기에 둔다.
"""

from uuid import uuid4

from interview.evidence.retrieval import search_evidence
from interview.llm import get_llm
from interview.schemas.question import Difficulty, Question, QuestionKind
from interview.strategy import prompts


def generate_question(topic: str, difficulty: Difficulty) -> Question:
    """주제 + 난이도로 일반 질문 생성.

    TODO(담당 B):
      - search_evidence(topic=...) 로 근거 chunk 조회
      - prompts.QUESTION_GEN_SYSTEM + 근거로 LLM 호출
      - linked_evidence 에 사용한 chunk_id 기록
    """
    chunks = search_evidence(query=topic, topic=topic)

    # TODO(담당 B): 실제 구현에서는 get_llm(temperature=0.4)로
    # prompts.QUESTION_GEN_SYSTEM + chunks를 전달해 질문을 생성한다.
    _ = get_llm
    _ = prompts

    return Question(
        question_id=_question_id("main"),
        text=f"{topic}에 대해 핵심 개념과 경험을 함께 설명해주세요.",
        topic=topic,
        difficulty=difficulty,
        kind="main",
        evidence_ids=[chunk.chunk_id for chunk in chunks],
    )


def generate_follow_up(topic: str, missing_keywords: list[str]) -> Question:
    """답변이 얕을 때 부족한 키워드를 끌어내는 꼬리 질문 생성.

    설계 문서 예시:
      missing=["fetch join","지연 로딩"] → "N+1 이 지연 로딩 상황에서 어떻게
      발생하는지 설명하고, fetch join 으로 어떻게 줄이는지 말해보세요."

    TODO(담당 B): prompts.FOLLOW_UP_SYSTEM + 근거 + missing_keywords 로 생성
    """
    keyword_text = ", ".join(missing_keywords) if missing_keywords else "빠진 핵심 개념"
    return Question(
        question_id=_question_id("follow-up"),
        text=f"{topic} 답변에서 {keyword_text} 부분을 조금 더 구체적으로 설명해주세요.",
        topic=topic,
        difficulty="medium",
        kind="follow_up",
    )


def generate_hint(topic: str) -> Question:
    """막혔을 때 실마리를 주는 힌트성 질문 생성.

    TODO(담당 B): prompts.HINT_SYSTEM 사용
    """
    return Question(
        question_id=_question_id("hint"),
        text=f"{topic}의 정의, 사용 이유, 주의할 점 순서로 떠올려볼까요?",
        topic=topic,
        difficulty="easy",
        kind="hint",
    )


def _question_id(prefix: QuestionKind | str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"
