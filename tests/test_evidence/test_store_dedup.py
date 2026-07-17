"""Evidence Top-K 결과의 중복 제거 동작을 검증한다."""

from interview.evidence.store import EvidenceStore
from interview.schemas.evidence import EvidenceChunk, SourceType


def _chunk(chunk_id: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        source_type=SourceType.GITHUB,
        source_url="https://github.com/example/repository",
        topic="DTO 구현",
        confidence=1.0,
    )


def test_deduplicate_chunks_removes_nearly_identical_text() -> None:
    """공백과 대소문자만 다른 거의 같은 Evidence는 하나만 남긴다."""

    store = EvidenceStore.__new__(EvidenceStore)
    chunks = [
        _chunk(
            "chunk-1",
            "private Long userId;\nprivate String name;",
        ),
        _chunk(
            "chunk-2",
            "  PRIVATE Long userId;   private String name;  ",
        ),
        _chunk(
            "chunk-3",
            "private Long boardCount;",
        ),
    ]

    result = store._deduplicate_chunks(chunks)

    assert [chunk.chunk_id for chunk in result] == [
        "chunk-1",
        "chunk-3",
    ]


def test_deduplicate_chunks_keeps_distinct_text() -> None:
    """서로 다른 Evidence는 입력 순서대로 모두 유지한다."""

    store = EvidenceStore.__new__(EvidenceStore)
    chunks = [
        _chunk("chunk-1", "private Long userId;"),
        _chunk("chunk-2", "private Long boardCount;"),
        _chunk("chunk-3", "private Status status;"),
    ]

    result = store._deduplicate_chunks(chunks)

    assert [chunk.chunk_id for chunk in result] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
