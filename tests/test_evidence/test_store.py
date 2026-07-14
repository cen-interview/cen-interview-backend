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
    store = EvidenceStore()

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
    store = EvidenceStore()
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
    store = EvidenceStore()
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
    store = EvidenceStore()
    store.add_chunks([_chunk("default-jpa-1", "JPA", 0.5)])

    results = store.query("JPA", topic="JPA")
    coverage = store.build_coverage_map()

    assert [chunk.chunk_id for chunk in results] == ["default-jpa-1"]
    assert coverage.topic_coverage["JPA"].chunk_count == 1


def test_chroma_backend_embeds_and_upserts_chunks() -> None:
    """chroma backend는 add_chunks에서 문서를 임베딩하고 metadata와 함께 upsert한다."""

    class FakeEmbeddings:
        """임베딩 호출 입력을 기록하는 fake embedding client."""

        def __init__(self) -> None:
            self.documents: list[list[str]] = []

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """문서 임베딩을 고정 벡터로 반환한다."""
            self.documents.append(texts)
            return [[1.0, 0.0], [0.0, 1.0]]

    class FakeCollection:
        """Chroma collection의 upsert 호출을 기록한다."""

        def __init__(self) -> None:
            self.upsert_calls: list[dict] = []

        def upsert(self, **kwargs) -> None:
            """upsert 인자를 기록한다."""
            self.upsert_calls.append(kwargs)

    embeddings = FakeEmbeddings()
    collection = FakeCollection()
    store = EvidenceStore(backend="chroma", embedding_client=embeddings)
    store._collection = collection
    chunks = [
        _chunk("chunk-1", "JPA", 0.8, text="JPA N+1 해결 근거"),
        _chunk("chunk-2", "Redis", 0.7, text="Redis 캐시 적용 근거"),
    ]

    store.add_chunks(chunks, user_id="user-1")

    assert embeddings.documents == [["JPA N+1 해결 근거", "Redis 캐시 적용 근거"]]
    assert len(collection.upsert_calls) == 1
    call = collection.upsert_calls[0]
    assert call["ids"] == ["user-1:chunk-1", "user-1:chunk-2"]
    assert call["documents"] == ["JPA N+1 해결 근거", "Redis 캐시 적용 근거"]
    assert call["embeddings"] == [[1.0, 0.0], [0.0, 1.0]]
    assert call["metadatas"][0]["user_id"] == "user-1"
    assert call["metadatas"][0]["chunk_id"] == "chunk-1"
    assert call["metadatas"][0]["topic"] == "JPA"


def test_chroma_backend_query_restores_evidence_chunks() -> None:
    """chroma backend query는 query embedding으로 검색하고 EvidenceChunk를 복원한다."""

    class FakeEmbeddings:
        """query 임베딩 호출을 기록하는 fake embedding client."""

        def __init__(self) -> None:
            self.queries: list[str] = []

        def embed_query(self, query: str) -> list[float]:
            """query 임베딩을 고정 벡터로 반환한다."""
            self.queries.append(query)
            return [0.5, 0.5]

    class FakeCollection:
        """Chroma query 결과를 반환하는 fake collection."""

        def __init__(self) -> None:
            self.query_calls: list[dict] = []

        def query(self, **kwargs) -> dict:
            """query 인자를 기록하고 저장된 청크 metadata를 반환한다."""
            self.query_calls.append(kwargs)
            return {
                "documents": [["JPA N+1 해결 근거"]],
                "metadatas": [
                    [
                        {
                            "chunk_id": "chunk-1",
                            "source_type": "notion",
                            "source_url": "https://notion.so/chunk-1",
                            "topic": "JPA",
                            "doc_type": "study",
                            "week": -1,
                            "date": "",
                            "confidence": 0.8,
                        }
                    ]
                ],
            }

    embeddings = FakeEmbeddings()
    collection = FakeCollection()
    store = EvidenceStore(backend="chroma", embedding_client=embeddings)
    store._collection = collection

    results = store.query("N+1", topic="JPA", user_id="user-1")

    assert embeddings.queries == ["N+1"]
    assert collection.query_calls[0]["query_embeddings"] == [[0.5, 0.5]]
    assert collection.query_calls[0]["where"] == {
        "$and": [{"user_id": "user-1"}, {"topic": "JPA"}]
    }
    assert [chunk.chunk_id for chunk in results] == ["chunk-1"]
    assert results[0].text == "JPA N+1 해결 근거"
