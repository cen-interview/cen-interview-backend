"""질문 생성 로직 (LLM + 근거).

근거를 조회해 실제 질문 문장을 만든다. agent.py 는 이 함수들을 배선만 하고,
실제 "어떻게 생성하느냐"는 여기에 둔다.
"""

from uuid import uuid4

from interview.evidence.retrieval import search_evidence
from interview.schemas.question import Difficulty, Question,QuestionKind


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



def generate_follow_up(topic: str, target: str | None = None) -> Question:
    """추가 확인 가능한 요소에 대한 꼬리 질문 생성."""

    probe = target or "추가로 설명할 수 있는 부분"
    evidence_chunks = search_evidence(query=probe, topic=topic)

    return Question(
        question_id=str(uuid4()),
        text=f"{topic} 답변에서 '{probe}' 부분을 조금 더 구체적으로 설명해 주세요.",
        topic=topic,
        difficulty=Difficulty.EASY,
        kind=QuestionKind.FOLLOW_UP,
        evidence_ids=[chunk.chunk_id for chunk in evidence_chunks],
        parent_question_id=None,
    )


def generate_challenge(topic: str, target: str | None = None) -> Question:
    """오개념이나 논리적 허점을 검증하는 압박 질문 생성."""

    probe = target or "답변의 논리적 근거"
    evidence_chunks = search_evidence(query=probe, topic=topic)

    return Question(
        question_id=str(uuid4()),
        text=f"{topic} 답변에서 '{probe}' 부분이 조금 더 검증이 필요합니다. 그 근거를 다시 설명해 주시겠어요?",
        topic=topic,
        difficulty=Difficulty.EASY,
        kind=QuestionKind.CHALLENGE,
        evidence_ids=[chunk.chunk_id for chunk in evidence_chunks],
        parent_question_id=None,
    )


def generate_confirm_positive(topic: str, target: str | None = None) -> Question:
    """답변이 대체로 맞지만 범위나 사실관계를 확인하는 긍정 확인 질문 생성."""

    probe = target or "답변의 적용 범위"
    evidence_chunks = search_evidence(query=probe, topic=topic)

    return Question(
        question_id=str(uuid4()),
        text=f"좋습니다. {topic}에서 말씀하신 '{probe}' 부분은 실제 프로젝트에서도 그렇게 적용하신 건가요?",
        topic=topic,
        difficulty=Difficulty.EASY,
        kind=QuestionKind.CONFIRM_POSITIVE,
        evidence_ids=[chunk.chunk_id for chunk in evidence_chunks],
        parent_question_id=None,
    )


def generate_confirm_negative(topic: str, target: str | None = None) -> Question:
    """Evidence 또는 이전 답변과 충돌하는 내용을 확인하는 부정 확인 질문 생성."""

    probe = target or "답변과 근거가 다른 부분"
    evidence_chunks = search_evidence(query=probe, topic=topic)

    return Question(
        question_id=str(uuid4()),
        text=f"{topic}에 대해 말씀하신 내용 중 '{probe}' 부분이 기존 근거와 다르게 보입니다. 다른 프로젝트나 계획 단계였던 부분일까요?",
        topic=topic,
        difficulty=Difficulty.EASY,
        kind=QuestionKind.CONFIRM_NEGATIVE,
        evidence_ids=[chunk.chunk_id for chunk in evidence_chunks],
        parent_question_id=None,
    )


def generate_trap(topic: str, target: str | None = None) -> Question:
    """헷갈리기 쉬운 개념 구분을 확인하는 함정 질문 생성."""

    probe = target or "헷갈리기 쉬운 개념"
    evidence_chunks = search_evidence(query=probe, topic=topic)

    return Question(
        question_id=str(uuid4()),
        text=f"{topic}에서 '{probe}'와 비슷해 보이지만 다른 개념이 있다면 어떻게 구분하시겠어요?",
        topic=topic,
        difficulty=Difficulty.EASY,
        kind=QuestionKind.TRAP,
        evidence_ids=[chunk.chunk_id for chunk in evidence_chunks],
        parent_question_id=None,
    )