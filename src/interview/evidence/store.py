"""evidence_store: 벡터 DB 래퍼.

청크를 임베딩해 적재하고, 쿼리로 유사 청크를 검색한다. Postgres + pgvector 를
쓰되 이 파일 안에만 DB 의존성을 가둬서 나중에 교체하기 쉽게 한다.

  - 적재(add_chunks): 인덱싱 파이프라인이 면접 전 1회 호출
  - 검색(query): Retrieval Tool(retrieval.py)이 런타임에 호출
"""

from interview.config import settings
from interview.schemas.evidence import CoverageMap, EvidenceChunk, TopicCoverage


class EvidenceStore:
    def __init__(self, database_url: str | None = None) -> None:
        """
        TODO(담당 A):
            - psycopg.connect(self.database_url) + CREATE EXTENSION vector
            - embedding_dimensions 크기의 vector 컬럼과 ivfflat 인덱스 초기화
            - embedding_model 로 생성한 벡터를 저장/검색에 사용
        """
        self.database_url = database_url or settings.database_url
        self.embedding_model = settings.embedding_model
        self.embedding_dimensions = settings.embedding_dimensions
        self._conn = None

        # [Stub 전용] 벡터 DB 대신 인메모리 리스트로 흉내낸다.
        self._chunks: list[EvidenceChunk] = []

    def add_chunks(self, chunks: list[EvidenceChunk]) -> None:
        """청크를 임베딩해 저장. 메타데이터도 함께 저장해 검색 후 복원한다.

        TODO(담당 A):
            - settings.embedding_model 로 chunk.text 임베딩 생성
            - embedding_dimensions 와 DB vector 컬럼 차원 일치 검증
            - INSERT ... ON CONFLICT (chunk_id) DO UPDATE
        """
        # [현재 Stub 작동] 임베딩 없이 그대로 적재
        self._chunks.extend(chunks)

    def query(self, query: str, topic: str | None = None, k: int = 5) -> list[EvidenceChunk]:
        """유사 청크 top-k 반환. topic 이 있으면 메타데이터 필터로 좁힌다.

        TODO(담당 A):
            - settings.embedding_model 로 query 임베딩 생성
            - pgvector `<->` 거리 연산자로 ORDER BY
            - topic 있으면 WHERE 절로 필터
            - EvidenceChunk 복원
        """
        # [현재 Stub 작동] 유사도 검색 대신 topic 일치 필터 + 앞에서부터 k개
        candidates = [c for c in self._chunks if topic is None or c.topic == topic]
        if candidates:
            return candidates[:k]
        if self._chunks:
            return self._chunks[:k]
        # 아직 인덱싱 전이어도 호출부(Strategy/Assessment)가 죽지 않도록 더미 1건 반환
        return [
            EvidenceChunk(
                chunk_id="stub_chunk_1",
                text=f"[Stub] '{query}' 관련 근거 예시 텍스트입니다.",
                source_type="notion",
                source_url="https://notion.so/stub",
                topic=topic or "General",
                confidence=0.5,
            )
        ]

    def build_coverage_map(self) -> CoverageMap:
        """저장된 청크들의 주제별 근거 커버리지를 집계한다.

        각 EvidenceChunk 를 topic 기준으로 묶고, 주제별 confidence 평균과
        청크 수를 계산한다. Strategy 는 이 결과를 보고 질문 주제를 고른다.

        Returns:
            주제별 평균 신뢰도와 근거 청크 수를 담은 CoverageMap.
        """
        # [현재 Stub 작동] topic 별 confidence 단순 평균
        by_topic: dict[str, list[float]] = {}
        for c in self._chunks:
            by_topic.setdefault(c.topic, []).append(c.confidence)
        return CoverageMap(
            topic_coverage={
                topic: TopicCoverage(
                    confidence=sum(confidences) / len(confidences),
                    chunk_count=len(confidences),
                )
                for topic, confidences in by_topic.items()
            },
            updated_at=None,
        )


# 런타임 공용 단일 인스턴스. retrieval.py 가 이걸 연다.
_store: EvidenceStore | None = None


def get_store() -> EvidenceStore:
    global _store
    if _store is None:
        _store = EvidenceStore()
    return _store
