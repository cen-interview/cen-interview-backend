"""테스트 공용 픽스처.

핵심 협업 패턴: 남의 에이전트가 아직 없어도, 그 에이전트의 '계약(schemas)'에
맞는 가짜(mock)를 만들어 내 코드를 독립적으로 테스트한다.
"""

import pytest

from interview.schemas.evidence import EvidenceChunk
from interview.schemas.question import Question
from interview.schemas.signals import AnswerQualitySignal

# 주의: Difficulty/QuestionKind/AnswerQuality/SourceType 은 schemas/ 에서
# Literal["a","b",...] 타입 별칭이지 Enum 이 아니다. Difficulty.MEDIUM 처럼
# 점 접근하면 AttributeError 가 난다 — 그냥 문자열 리터럴로 쓴다.


@pytest.fixture
def sample_question() -> Question:
    return Question(
        question_id="q1",
        text="JPA N+1 문제를 설명하세요.",
        topic="JPA",
        difficulty="medium",
        kind="main",
        evidence_ids=["c1"],
    )


@pytest.fixture
def shallow_signal() -> AnswerQualitySignal:
    return AnswerQualitySignal(
        question_id="q1",
        quality="shallow",
        missing_keywords=["fetch join", "지연 로딩"],
    )


@pytest.fixture
def sample_chunk() -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id="c1",
        text="N+1 문제는 연관관계 지연 로딩에서 쿼리가 N번 추가 발생...",
        source_url="https://notion.so/...",
        source_type="notion",
        topic="JPA",
        confidence=0.8,
    )
