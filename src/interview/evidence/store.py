"""evidence_store: 벡터 DB 래퍼.

청크를 임베딩해 적재하고, 쿼리로 유사 청크를 검색한다. chromadb 로 시작하되
이 파일 안에만 DB 의존성을 가둬서 나중에 교체하기 쉽게 한다.

  - 적재(add_chunks): 인덱싱 파이프라인이 면접 전 1회 호출
  - 검색(query): Retrieval Tool(retrieval.py)이 런타임에 호출
"""

from interview.config import settings
from interview.schemas.evidence import EvidenceChunk


class EvidenceStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or settings.evidence_store_path
        # TODO(담당 A): chromadb PersistentClient + 컬렉션 초기화
        self._client = None

    def add_chunks(self, chunks: list[EvidenceChunk]) -> None:
        """청크를 임베딩해 저장. 메타데이터도 함께 저장해 검색 후 복원한다.

        TODO(담당 A): 임베딩 → upsert (chunk_id 를 DB id 로)
        """
        raise NotImplementedError

    def query(
        self, query: str, topic: str | None = None, k: int = 5
    ) -> list[EvidenceChunk]:
        """유사 청크 top-k 반환. topic 이 있으면 메타데이터 필터로 좁힌다.

        TODO(담당 A): 벡터 검색 + 메타데이터 필터 → EvidenceChunk 복원
        """
        raise NotImplementedError

    def build_coverage_map(self) -> CoverageMap:
        """저장된 청크들의 주제별 신뢰도를 집계해 커버리지 맵 생성.

        TODO(담당 A): topic 별 confidence 평균 → CoverageMap
        """
        raise NotImplementedError


# 런타임 공용 단일 인스턴스. retrieval.py 가 이걸 연다.
_store: EvidenceStore | None = None


def get_store() -> EvidenceStore:
    global _store
    if _store is None:
        _store = EvidenceStore()
    return _store
