"""EvidenceStore 인메모리 POC 동작 검증."""

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


def test_rejects_unsupported_backend() -> None:
    """지원하지 않는 저장소 backend 설정은 거부한다."""
    try:
        EvidenceStore(backend="unsupported")
    except ValueError as exc:
        assert "pgvector" in str(exc)
    else:
        raise AssertionError("unsupported backend must be rejected")
