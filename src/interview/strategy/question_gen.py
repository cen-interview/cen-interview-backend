"""질문 생성 로직 (LLM + 근거).

근거를 조회해 실제 질문 문장을 만든다. agent.py 는 이 함수들을 배선만 하고,
실제 "어떻게 생성하느냐"는 여기에 둔다.
"""

from interview.evidence.retrieval import search_evidence
from interview.llm import get_llm
from interview.schemas.question import Difficulty, Question, QuestionKind  # noqa: F401 (TODO 담당 B: 실제 구현 시 사용)
from interview.strategy import prompts  # noqa: F401 (TODO 담당 B: 실제 LLM 호출 시 사용)


def generate_question(topic: str, difficulty: Difficulty) -> Question:
    """주제 + 난이도로 일반 질문 생성.

    TODO(담당 B):
      - search_evidence(topic=...) 로 근거 chunk 조회
      - prompts.QUESTION_GEN_SYSTEM + 근거로 LLM 호출
      - linked_evidence 에 사용한 chunk_id 기록
    """
    _ = search_evidence(query=topic, topic=topic)
    _ = get_llm(temperature=0.4)
    raise NotImplementedError


def generate_follow_up(topic: str, missing_keywords: list[str]) -> Question:
    """답변이 얕을 때 부족한 키워드를 끌어내는 꼬리 질문 생성.

    설계 문서 예시:
      missing=["fetch join","지연 로딩"] → "N+1 이 지연 로딩 상황에서 어떻게
      발생하는지 설명하고, fetch join 으로 어떻게 줄이는지 말해보세요."

    TODO(담당 B): prompts.FOLLOW_UP_SYSTEM + 근거 + missing_keywords 로 생성
    """
    raise NotImplementedError


def generate_hint(topic: str) -> Question:
    """막혔을 때 실마리를 주는 힌트성 질문 생성.

    TODO(담당 B): prompts.HINT_SYSTEM 사용
    """
    raise NotImplementedError
