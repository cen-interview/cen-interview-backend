"""Evidence 도메인 최소 동작 테스트."""

from interview.evidence.chunking import chunk
from interview.evidence.extract import extract_evidence
from interview.evidence.indexing import build_index
from interview.evidence.sources import RawDoc
from interview.evidence.store import EvidenceStore
from interview.schemas.evidence import CoverageMap


def test_chunk_splits_long_text_and_preserves_metadata(sample_chunk):
    chunks = chunk(
        [sample_chunk.model_copy(update={"text": "a" * 25})],
        max_chars=10,
    )

    assert [item.chunk_id for item in chunks] == ["c1:0", "c1:1", "c1:2"]
    assert [len(item.text) for item in chunks] == [10, 10, 5]
    assert all(item.topic == sample_chunk.topic for item in chunks)
    assert all(item.source_url == sample_chunk.source_url for item in chunks)


def test_extract_evidence_maps_raw_doc_to_chunk():
    raw_doc = RawDoc(
        source_url="https://notion.so/page",
        source_type="notion",
        title="Spring 트랜잭션",
        raw_text="트랜잭션 전파 속성과 격리 수준을 정리한 문서입니다.",
        meta={
            "topic": "Spring",
            "doc_type": "notion_page",
            "confidence": 0.9,
        },
    )

    chunks = extract_evidence(raw_doc)

    assert len(chunks) == 1
    assert chunks[0].source_type == "notion"
    assert chunks[0].topic == "Spring"
    assert chunks[0].confidence == 0.9


def test_store_queries_by_topic(sample_chunk):
    store = EvidenceStore()
    store.add_chunks([sample_chunk])

    results = store.query("N+1", topic="JPA")

    assert results == [sample_chunk]


def test_store_builds_coverage_map(sample_chunk):
    store = EvidenceStore()
    store.add_chunks(
        [
            sample_chunk,
            sample_chunk.model_copy(
                update={"chunk_id": "c2", "topic": "JPA", "confidence": 0.6}
            ),
            sample_chunk.model_copy(
                update={"chunk_id": "c3", "topic": "Spring", "confidence": 0.3}
            ),
        ]
    )

    coverage = store.build_coverage_map()

    assert coverage.topic_confidence["JPA"] == 0.7
    assert coverage.weak_topics() == ["Spring"]
    assert coverage.strong_topics() == ["JPA"]


def test_build_index_returns_coverage_map():
    coverage = build_index(
        notion_link="https://notion.so/example",
        github_links=["https://github.com/example/cen-interview-backend"],
    )

    assert isinstance(coverage, CoverageMap)
    assert coverage.topic_confidence
