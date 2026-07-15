"""질문 생성 로직 (LLM + 근거).

근거를 조회해 실제 질문 문장을 만든다. agent.py 는 이 함수들을 배선만 하고,
실제 "어떻게 생성하느냐"는 여기에 둔다.
"""

from uuid import uuid4

from pydantic import BaseModel, Field

from interview.evidence.retrieval import search_evidence
from interview.llm.client import get_llm
from interview.llm.logging import log_llm_error, log_llm_output
from interview.schemas.question import Difficulty, Question, QuestionCategory,QuestionKind
from interview.strategy.prompts import (
    QUESTION_GEN_SYSTEM,
    FOLLOW_UP_SYSTEM,
    CHALLENGE_SYSTEM,
    CONFIRM_POSITIVE_SYSTEM,
    CONFIRM_NEGATIVE_SYSTEM,
    TRAP_SYSTEM,
    HINT_SYSTEM
)
_EVIDENCE_CONFIDENCE_THRESHOLD = 0.4

_DERIVED_DIFFICULTY: dict[QuestionKind, Difficulty] = {
    QuestionKind.FOLLOW_UP: Difficulty.EASY,
    QuestionKind.CHALLENGE: Difficulty.MEDIUM,
    QuestionKind.CONFIRM_POSITIVE: Difficulty.EASY,
    QuestionKind.CONFIRM_NEGATIVE: Difficulty.EASY,
    QuestionKind.TRAP: Difficulty.MEDIUM,
}

_DERIVED_SYSTEM_PROMPTS: dict[QuestionKind, str] = {
    QuestionKind.FOLLOW_UP: FOLLOW_UP_SYSTEM,
    QuestionKind.CHALLENGE: CHALLENGE_SYSTEM,
    QuestionKind.CONFIRM_POSITIVE: CONFIRM_POSITIVE_SYSTEM,
    QuestionKind.CONFIRM_NEGATIVE: CONFIRM_NEGATIVE_SYSTEM,
    QuestionKind.TRAP: TRAP_SYSTEM,
}
class GeneratedQuestion(BaseModel):
    """LLM이 생성하는 질문의 구조화 출력."""
    text: str = Field(description="생성된 질문 문장. 반드시 하나의 질문만 담는다.")
    category: QuestionCategory = Field(
        description="질문 카테고리: technical(기술개념), project(프로젝트구현)중 하나."
    )

class GeneratedDerivedQuestion(BaseModel):
    """LLM이 생성하는 파생 질문의 구조화 출력."""
    text: str = Field(description="생성된 질문 문장. 반드시 하나의 질문만 담는다.")
    category: QuestionCategory = Field(
        description="질문 카테고리: technical(기술개념), project(프로젝트구현) 중 하나. "
        "부모 질문의 맥락과 다를 수 있다 (예: 프로젝트 질문에서 파생된 기술개념 확인 질문)."
    )

class GeneratedHint(BaseModel):
    """LLM이 생성하는 힌트의 구조화 출력."""
    text: str = Field(description="정답을 직접 알려주지 않는 힌트 문장. 한두 문장으로 짧게.")

def _generate_derived_question(
    kind: QuestionKind,
    topic: str,
    parent_question_id: str,
    target: str | None,
    answer_excerpt: str | None,
    rationale: list[str] | None = None,
    user_id: str | None = None,
) -> Question:
    """파생 질문(follow_up/challenge/confirm_positive/confirm_negative/trap) 공통 생성 로직."""
    probe = target or "답변에서 더 확인이 필요한 부분"
    evidence_chunks = search_evidence(query=probe, topic=topic, k=5, user_id=user_id)
    reliable_chunks = [c for c in evidence_chunks if c.confidence >= _EVIDENCE_CONFIDENCE_THRESHOLD]

    context = (
        "\n".join(f"- {c.text}" for c in reliable_chunks)
        if reliable_chunks
        else "(관련 근거 없음)"
    )
    excerpt_block = f'"{answer_excerpt}"' if answer_excerpt else "(답변 발췌 없음)"

    rationale_block = (
        "\n".join(f"- {r}" for r in rationale)
        if rationale
        else "(없음)"
    )

    user_prompt = f"""\
주제: {topic}
파고들 대상(target): {probe}

사용자의 직전 답변 중 관련 발췌:
{excerpt_block}

이 부분이 문제로 판단된 이유:
{rationale_block}

근거:
{context}
"""

    system_prompt = _DERIVED_SYSTEM_PROMPTS[kind]
    difficulty = _DERIVED_DIFFICULTY[kind]
    llm = get_llm(temperature=0.6)
    structured_llm = llm.with_structured_output(GeneratedDerivedQuestion)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = result.text
        category = result.category
        log_llm_output(
            "DERIVED_QUESTION_GENERATION",
            result,
            metadata={
                "topic": topic,
                "question_kind": kind.value,
                "difficulty": difficulty.value,
                "parent_question_id": parent_question_id,
                "target": probe,
                "evidence_ids": [chunk.chunk_id for chunk in reliable_chunks],
            },
            input_data={
                "user_prompt": user_prompt,
                "answer_excerpt": answer_excerpt,
            },
        )
    except Exception as exc:
        text = f"{topic} 답변에서 '{probe}' 부분을 조금 더 설명해 주시겠어요?"
        category = None
        log_llm_error(
            "DERIVED_QUESTION_GENERATION",
            exc,
            metadata={
                "topic": topic,
                "question_kind": kind.value,
                "difficulty": difficulty.value,
                "parent_question_id": parent_question_id,
                "target": probe,
                "evidence_ids": [chunk.chunk_id for chunk in reliable_chunks],
            },
            fallback={"text": text, "category": category},
            input_data={
                "user_prompt": user_prompt,
                "answer_excerpt": answer_excerpt,
            },
        )

    return Question(
        question_id=str(uuid4()),
        text=text,
        topic=topic,
        difficulty=difficulty,
        kind=kind,
        category=category,
        evidence_ids=[c.chunk_id for c in reliable_chunks],
        parent_question_id=parent_question_id,
    )

def generate_question(
    topic: str,
    difficulty: Difficulty,
    asked_question_texts: list[str] | None = None,
    user_id: str | None = None,
) -> Question:
    """주제 + 난이도로 일반 질문 생성.

    근거를 조회해 QUESTION_GEN_SYSTEM 프롬프트와 함께 LLM에 전달하고,
    구조화된 출력(text, category)을 받아 Question으로 구성한다.

    Args:
        topic: 질문 주제.
        difficulty: 질문 난이도.
        asked_question_texts: 이미 출제된 질문 문장들 (중복 방지용).
            LLM에게 "이런 질문은 이미 했으니 겹치지 않게 하라"고 전달한다.
        user_id: 사용자 id. 현재는 None 가능.


    Returns:
        kind=MAIN인 Question. evidence_ids에 실제 조회된 근거 chunk_id가 담긴다.

    TODO(담당 B):
      - prompts.QUESTION_GEN_SYSTEM + 근거로 LLM 호출
      - linked_evidence 에 사용한 chunk_id 기록
    """
    evidence_chunks = search_evidence(query=topic, topic=topic, user_id=user_id)

    reliable_chunks = [c for c in evidence_chunks if c.confidence >= _EVIDENCE_CONFIDENCE_THRESHOLD]

    context = (
        "\n".join(f"- {c.text}" for c in reliable_chunks)
        if reliable_chunks
        else "(관련 근거 없음. 근거를 인용하지 말고 일반적인 개념 질문으로 만들 것)"
    )

    asked_block = (
        "\n".join(f"- {t}" for t in asked_question_texts)
        if asked_question_texts
        else "(없음)"
    )

    user_prompt = f"""\
주제: {topic}
난이도: {difficulty.value}

근거:
{context}

이미 출제한 질문 (아래와 겹치지 않는 새로운 질문을 만들 것):
{asked_block}
"""

    llm = get_llm(temperature=0.6)
    structured_llm = llm.with_structured_output(GeneratedQuestion)
    try :

        result = structured_llm.invoke(
            [
                {"role": "system", "content": QUESTION_GEN_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )
        log_llm_output(
            "MAIN_QUESTION_GENERATION",
            result,
            metadata={
                "topic": topic,
                "difficulty": difficulty.value,
                "question_kind": QuestionKind.MAIN.value,
                "evidence_ids": [chunk.chunk_id for chunk in reliable_chunks],
            },
            input_data={"user_prompt": user_prompt},
        )
    except Exception as exc:
        fallback = _fallback_question(
            topic,
            difficulty,
            [chunk.chunk_id for chunk in reliable_chunks],
        )
        log_llm_error(
            "MAIN_QUESTION_GENERATION",
            exc,
            metadata={
                "topic": topic,
                "difficulty": difficulty.value,
                "question_kind": QuestionKind.MAIN.value,
                "evidence_ids": fallback.evidence_ids,
            },
            fallback=fallback,
            input_data={"user_prompt": user_prompt},
        )
        return fallback

    return Question(
        question_id=str(uuid4()),
        text=result.text,
        topic=topic,
        difficulty=difficulty,
        kind=QuestionKind.MAIN,
        category=result.category,
        evidence_ids=[chunk.chunk_id for chunk in reliable_chunks],
        parent_question_id=None,
    )


def generate_follow_up(
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
        rationale: list[str] | None = None,
        user_id: str | None = None,
    ) -> Question:
    """추가 확인 가능한 요소에 대한 꼬리 질문 생성."""
    return _generate_derived_question(
        QuestionKind.FOLLOW_UP, topic, parent_question_id, target, answer_excerpt, rationale, user_id
    )


def generate_challenge(
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
        rationale: list[str] | None = None,
        user_id: str | None = None,
    )-> Question:
    """오개념이나 논리적 허점을 검증하는 압박 질문 생성."""
    return _generate_derived_question(
        QuestionKind.CHALLENGE, topic, parent_question_id, target, answer_excerpt, rationale, user_id
    )


def generate_confirm_positive(
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
        rationale: list[str] | None = None,
        user_id: str | None = None,

    ) -> Question:
    """답변이 대체로 맞지만 범위나 사실관계를 확인하는 긍정 확인 질문 생성."""
    return _generate_derived_question(
        QuestionKind.CONFIRM_POSITIVE, topic, parent_question_id, target, answer_excerpt, rationale, user_id
    )

def generate_confirm_negative(
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
        rationale: list[str] | None = None,
        user_id: str | None = None,
    ) -> Question:
    """Evidence 또는 이전 답변과 충돌하는 내용을 확인하는 부정 확인 질문 생성."""
    return _generate_derived_question(
        QuestionKind.CONFIRM_NEGATIVE, topic, parent_question_id, target, answer_excerpt, rationale, user_id
    )


def generate_trap(
    topic: str,
    parent_question_id: str,
    target: str | None = None,
    answer_excerpt: str | None = None,
    rationale: list[str] | None = None,
    user_id: str | None = None,
    ) -> Question:
    """헷갈리기 쉬운 개념 구분을 확인하는 함정 질문 생성."""
    return _generate_derived_question(
        QuestionKind.TRAP, topic, parent_question_id, target, answer_excerpt, rationale, user_id
    )

def generate_hint(
    question: Question,
    target: str | None = None,
    answer_excerpt: str | None = None,
    user_id: str | None = None,
    ) -> Question:
    """침묵 등으로 사용자가 답변을 못 할 때 호출하는 힌트 생성.

    정답을 알려주지 않고 접근 방향만 제시한다 (6단계에서 구현 예정).

    Args:
        question: 힌트를 줄 대상이 되는 원래 질문.
        target: 힌트를 어느 부분에 집중할지 (선택).
        answer_excerpt: 사용자의 직전 답변 중 인용할 부분 (선택).
            완전 침묵이면 None. 답변은 했지만 방향이 틀린 경우 참고용으로 전달.
    Returns:
        kind=HINT인 Question. parent_question_id는 원래 question의 ID.
    """

    probe = target or "질문의 핵심 개념"
    evidence_chunks = search_evidence(query=probe, topic=question.topic, k=5, user_id=user_id)
    reliable_chunks = [c for c in evidence_chunks if c.confidence >= _EVIDENCE_CONFIDENCE_THRESHOLD]

    context = (
        "\n".join(f"- {c.text}" for c in reliable_chunks)
        if reliable_chunks
        else "(관련 근거 없음)"
    )
    excerpt_block = f'"{answer_excerpt}"' if answer_excerpt else "(답변 없음, 완전 침묵 상태)"

    user_prompt = f"""\
원래 질문: {question.text}
힌트가 필요한 부분(target): {probe}

지원자의 답변 상태:
{excerpt_block}

근거 (키워드 1개까지만 활용):
{context}
"""
    
    llm = get_llm(temperature=0.6)
    structured_llm = llm.with_structured_output(GeneratedHint)

    try :
        result = structured_llm.invoke(
            [
                {"role": "system", "content": HINT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = result.text
        log_llm_output(
            "HINT_GENERATION",
            result,
            metadata={
                "topic": question.topic,
                "question_kind": QuestionKind.HINT.value,
                "parent_question_id": question.question_id,
                "target": probe,
                "evidence_ids": [chunk.chunk_id for chunk in reliable_chunks],
            },
            input_data={
                "user_prompt": user_prompt,
                "answer_excerpt": answer_excerpt,
            },
        )
    except Exception as exc:
        text = f"힌트: {question.text}에 대해 생각할 때 '{probe}' 부분을 고려해 보세요."
        log_llm_error(
            "HINT_GENERATION",
            exc,
            metadata={
                "topic": question.topic,
                "question_kind": QuestionKind.HINT.value,
                "parent_question_id": question.question_id,
                "target": probe,
                "evidence_ids": [chunk.chunk_id for chunk in reliable_chunks],
            },
            fallback={"text": text},
            input_data={
                "user_prompt": user_prompt,
                "answer_excerpt": answer_excerpt,
            },
        )

    return Question(
        question_id=str(uuid4()),
        text=text,
        topic=question.topic,
        difficulty=question.difficulty,
        category=question.category,
        kind=QuestionKind.HINT,
        evidence_ids=[chunk.chunk_id for chunk in reliable_chunks],
        parent_question_id=question.question_id,
    )

def _fallback_question(topic: str, difficulty: Difficulty, evidence_ids: list[str]) -> Question:
    """LLM 호출 실패 시 사용하는 템플릿 질문."""
    return Question(
        question_id=str(uuid4()),
        text=f"{topic}에 대해 설명해 주세요.",
        topic=topic,
        difficulty=difficulty,
        kind=QuestionKind.MAIN,
        evidence_ids=evidence_ids,
        parent_question_id=None,
    )
