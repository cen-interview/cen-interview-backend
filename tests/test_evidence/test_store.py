"""EvidenceStore 인메모리 POC 동작 검증."""

import pytest

from interview.evidence.store import EvidenceStore
from interview.schemas.evidence import EvidenceChunk, SourceType


def _chunk(
    chunk_id: str,
    topic: str,
    confidence: float,
    text: str | None = None,
) -> EvidenceChunk:
    """테스트에서 사용할 수제 EvidenceChunk를 만든다."""
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text or f"{topic} 테스트 근거",
        source_type=SourceType.NOTION,
        source_url=f"https://notion.so/{chunk_id}",
        topic=topic,
        doc_type="study",
        confidence=confidence,
    )


def test_query_keeps_chunks_isolated_by_user_id() -> None:
    """사용자별 namespace에 저장된 청크가 서로 섞이지 않는지 확인한다."""
    store = EvidenceStore(backend="memory")

    store.add_chunks(
        [
            _chunk("u1-jpa-1", "JPA", 0.8),
            _chunk("u1-jpa-2", "JPA", 0.6),
            _chunk("u1-redis-1", "Redis", 0.9),
        ],
        user_id="user-1",
    )
    store.add_chunks([_chunk("u2-jpa-1", "JPA", 0.4)], user_id="user-2")

    user_1_results = store.query("JPA", topic="JPA", user_id="user-1")
    user_2_results = store.query("JPA", topic="JPA", user_id="user-2")

    assert [chunk.chunk_id for chunk in user_1_results] == ["u1-jpa-1", "u1-jpa-2"]
    assert [chunk.chunk_id for chunk in user_2_results] == ["u2-jpa-1"]


def test_query_filters_by_topic_inside_user_namespace() -> None:
    """같은 사용자 namespace 안에서 topic 필터가 동작하는지 확인한다."""
    store = EvidenceStore(backend="memory")
    store.add_chunks(
        [
            _chunk("u1-jpa-1", "JPA", 0.8),
            _chunk("u1-redis-1", "Redis", 0.9),
        ],
        user_id="user-1",
    )

    results = store.query("Redis", topic="Redis", user_id="user-1")

    assert [chunk.chunk_id for chunk in results] == ["u1-redis-1"]


def test_query_filters_by_ownership_before_limit() -> None:
    """ownership 조건에 맞는 청크 중에서 k개를 반환해야 한다."""
    store = EvidenceStore(backend="memory")
    context_chunks = [
        _chunk(f"context-{index}", "WebSocket", 0.8).model_copy(
            update={"ownership": "repo_context"}
        )
        for index in range(5)
    ]
    touched_chunks = [
        _chunk(f"touched-{index}", "WebSocket", 0.9).model_copy(
            update={"ownership": "user_touched"}
        )
        for index in range(3)
    ]
    store.add_chunks(context_chunks + touched_chunks, user_id="user-1")

    results = store.query(
        "WebSocket",
        topic="WebSocket",
        k=2,
        user_id="user-1",
        ownership="user_touched",
    )

    assert [chunk.chunk_id for chunk in results] == ["touched-0", "touched-1"]


@pytest.mark.parametrize(
    ("min_similarity", "expected_max_distance"),
    [
        (0.30, 0.70),
        (0.80, 0.20),
    ],
)
def test_pgvector_query_applies_configured_min_similarity(
    monkeypatch,
    min_similarity: float,
    expected_max_distance: float,
) -> None:
    """설정한 최소 유사도가 pgvector의 cosine distance 상한으로 변환되어야 한다."""
    captured_statements = []

    class FakeEmbeddings:
        def embed_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    class EmptyScalarResult:
        def all(self) -> list:
            return []

    class CapturingSession:
        def scalars(self, statement):
            captured_statements.append(statement)
            return EmptyScalarResult()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "interview.evidence.store.settings.evidence_min_similarity",
        min_similarity,
    )
    store = EvidenceStore(
        backend="pgvector",
        embedding_client=FakeEmbeddings(),
        session_factory=CapturingSession,
    )

    results = store.query("로그인 접근 제어", k=5, user_id="user-1")

    assert results == []
    assert len(captured_statements) == 1
    statement = captured_statements[0]
    similarity_clause = statement._where_criteria[-1]
    assert similarity_clause.right.value == pytest.approx(expected_max_distance)
    assert statement._limit_clause.value == 5


def test_embedding_batches_deduplicate_text_and_preserve_chunk_order(monkeypatch) -> None:
    """중복 본문은 한 번만 임베딩하고 결과는 원래 청크 순서로 복원한다."""
    calls: list[list[str]] = []

    class FakeEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return [[float(len(text))] for text in texts]

    monkeypatch.setattr(
        "interview.evidence.store.settings.evidence_embedding_batch_size",
        2,
    )
    monkeypatch.setattr(
        "interview.evidence.store.settings.evidence_embedding_concurrency",
        1,
    )
    store = EvidenceStore(backend="pgvector", embedding_client=FakeEmbeddings())
    chunks = [
        _chunk("one", "JPA", 0.8, text="same"),
        _chunk("two", "JPA", 0.8, text="different"),
        _chunk("three", "JPA", 0.8, text="same"),
    ]

    result = store._embed_chunk_texts(chunks)

    assert calls == [["same", "different"]]
    assert result == [[4.0], [9.0], [4.0]]


def test_build_coverage_map_aggregates_by_topic_per_user() -> None:
    """CoverageMap이 사용자별 topic confidence 평균과 chunk 수를 집계하는지 확인한다."""
    store = EvidenceStore(backend="memory")
    store.add_chunks(
        [
            _chunk("u1-jpa-1", "JPA", 0.8),
            _chunk("u1-jpa-2", "JPA", 0.6),
            _chunk("u1-redis-1", "Redis", 0.9),
        ],
        user_id="user-1",
    )
    store.add_chunks([_chunk("u2-jpa-1", "JPA", 0.2)], user_id="user-2")

    coverage = store.build_coverage_map(user_id="user-1")

    assert coverage.topic_coverage["JPA"].chunk_count == 2
    assert coverage.topic_coverage["JPA"].confidence == 0.7
    assert coverage.topic_coverage["Redis"].chunk_count == 1
    assert coverage.topic_coverage["Redis"].confidence == 0.9
    assert "u2-jpa-1" not in coverage.topic_coverage


def test_default_namespace_keeps_existing_calls_working() -> None:
    """user_id가 없는 기존 호출은 default namespace에서 저장과 조회가 가능해야 한다."""
    store = EvidenceStore(backend="memory")
    store.add_chunks([_chunk("default-jpa-1", "JPA", 0.5)])

    results = store.query("JPA", topic="JPA")
    coverage = store.build_coverage_map()

    assert [chunk.chunk_id for chunk in results] == ["default-jpa-1"]
    assert coverage.topic_coverage["JPA"].chunk_count == 1


def test_clear_user_sources_keeps_other_source_chunks() -> None:
    """GitHub 재인덱싱은 같은 사용자의 Notion 근거를 지우면 안 된다."""
    store = EvidenceStore(backend="memory")
    notion = _chunk("notion-1", "JPA", 0.8)
    github = _chunk("github-1", "Java", 0.7)
    github = github.model_copy(
        update={"source_type": SourceType.GITHUB, "file_path": "src/AuthService.java"}
    )
    store.add_chunks([notion, github], user_id="user-1")

    store.clear_user_sources({"github"}, user_id="user-1")

    assert [chunk.chunk_id for chunk in store.query("", user_id="user-1")] == ["notion-1"]


def test_pgvector_record_values_preserve_github_provenance() -> None:
    """pgvector 저장 입력과 복원 결과가 GitHub provenance를 보존해야 한다."""
    store = EvidenceStore()
    chunk = EvidenceChunk(
        chunk_id="github-1",
        text="public class AuthService {}",
        source_type=SourceType.GITHUB,
        source_url="https://github.com/example/project/blob/HEAD/src/AuthService.java",
        topic="java",
        doc_type="code",
        confidence=0.8,
        file_path="src/AuthService.java",
        language="java",
        ownership="user_touched",
        commit_count=3,
        last_commit_sha="a" * 40,
    )

    values = store._record_values(
        chunk=chunk,
        namespace="user-1",
        embedding=[0.1, 0.2],
    )

    class Record:
        """ORM 조회 결과와 같은 속성을 제공하는 테스트 대역."""

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    restored = store._chunk_from_record(Record(**values))

    assert restored.file_path == "src/AuthService.java"
    assert restored.ownership == "user_touched"
    assert restored.commit_count == 3
    assert restored.last_commit_sha == "a" * 40


def test_pgvector_record_values_preserve_repo_context_ownership() -> None:
    """사용자가 직접 변경하지 않은 프로젝트 문맥도 ownership을 보존한다."""
    store = EvidenceStore()
    chunk = EvidenceChunk(
        chunk_id="github-context-1",
        text="class WebSocketConfig {}",
        source_type=SourceType.GITHUB,
        source_url="https://github.com/example/project/blob/HEAD/src/WebSocketConfig.java",
        topic="웹소켓 실시간 통신",
        doc_type="code",
        confidence=0.6,
        file_path="src/WebSocketConfig.java",
        language="java",
        ownership="repo_context",
    )

    values = store._record_values(
        chunk=chunk,
        namespace="user-1",
        embedding=[0.1, 0.2],
    )

    assert values["ownership"] == "repo_context"
    assert values["commit_count"] == 0


def test_rejects_unsupported_backend() -> None:
    """지원하지 않는 저장소 backend 설정은 거부한다."""
    try:
        EvidenceStore(backend="unsupported")
    except ValueError as exc:
        assert "pgvector" in str(exc)
    else:
        raise AssertionError("unsupported backend must be rejected")
