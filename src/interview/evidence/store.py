"""evidence_store: Evidence 청크 저장소 경계.

최종 목표는 청크를 임베딩해 Postgres + pgvector 에 적재하고, 쿼리로 유사
청크를 검색하는 것이다. 현재 구현은 호출부 계약을 먼저 고정하기 위한
user_id namespace별 인메모리 저장소다.

  - 적재(add_chunks): 인덱싱 파이프라인이 면접 전 1회 호출
  - 검색(query): Retrieval Tool(retrieval.py)이 런타임에 호출
"""

from interview.config import settings
from interview.schemas.evidence import CoverageMap, EvidenceChunk, TopicCoverage

DEFAULT_TOP_K = 5

class EvidenceStore:
    """Evidence 청크 저장/검색을 담당하는 저장소 경계.

    현재는 Postgres + pgvector 전환 전 임시 구현으로 user_id namespace별
    인메모리 저장소를 사용한다. 호출부는 이 클래스의 메서드 계약만 의존하게
    해서 실제 DB 구현으로 바뀌어도 Strategy/Assessment 영향을 줄인다.
    """

    DEFAULT_NAMESPACE = "default"

    def __init__(self, database_url: str | None = None) -> None:
        """EvidenceStore를 초기화하고 현재 구현용 인메모리 저장소를 준비한다.

        Args:
            database_url: pgvector 저장소로 전환할 때 사용할 DB URL.
                None이면 전역 settings.database_url을 사용한다.

        TODO(담당 A):
            - psycopg.connect(self.database_url) + CREATE EXTENSION vector
            - embedding_dimensions 크기의 vector 컬럼과 ivfflat 인덱스 초기화
            - embedding_model 로 생성한 벡터를 저장/검색에 사용
        """
        self.database_url = database_url or settings.database_url
        self.embedding_model = settings.embedding_model
        self.embedding_dimensions = settings.embedding_dimensions
        self._conn = None

        # 현재 구현: 벡터 DB 대신 사용자별 인메모리 리스트에 청크를 보관한다.
        self._chunks_by_user: dict[str, list[EvidenceChunk]] = {}

    def add_chunks(
        self,
        chunks: list[EvidenceChunk],
        user_id: int | str | None = None,
    ) -> None:
        """청크를 저장한다.

        현재 구현은 임베딩 없이 메모리에 보관한다. 실제 DB 구현에서는 여기서
        chunk.text 임베딩을 만들고 메타데이터와 함께 upsert한다.

        Args:
            chunks: 저장할 EvidenceChunk 목록.
            user_id: 사용자별 저장 namespace를 선택하기 위한 사용자 ID.
                기존 호출처럼 None이면 기본 namespace에 저장한다.

        TODO(담당 A):
            - settings.embedding_model 로 chunk.text 임베딩 생성
            - embedding_dimensions 와 DB vector 컬럼 차원 일치 검증
            - INSERT ... ON CONFLICT (chunk_id) DO UPDATE
        """
        # 현재 구현: 임베딩 없이 namespace별 chunks 목록에 그대로 적재한다.
        namespace = self._namespace(user_id)
        self._chunks_by_user.setdefault(namespace, []).extend(chunks)

    def clear_user(self, user_id: int | str | None = None) -> None:
        """사용자 namespace에 저장된 기존 청크를 삭제한다.

        같은 사용자가 Notion/GitHub 링크를 다시 등록하면 이전 인덱싱 결과가
        새 결과와 섞이지 않아야 한다. 실제 DB 구현에서는 이 메서드가
        user_id 조건 delete 또는 collection drop 역할을 한다.
        """
        namespace = self._namespace(user_id)
        self._chunks_by_user[namespace] = []

    def query(
        self,
        query: str,
        topic: str | None = None,
        k: int = DEFAULT_TOP_K,
        user_id: int | str | None = None,
    ) -> list[EvidenceChunk]:
        """유사 청크 top-k 반환. topic 이 있으면 메타데이터 필터로 좁힌다.

        Args:
            query: 검색할 질문 또는 주제 문장.
            topic: 특정 기술 주제로 검색 범위를 좁히기 위한 선택 필터.
            k: 반환할 최대 chunk 수.
            user_id: 사용자별 저장 namespace를 선택하기 위한 사용자 ID.
                기존 호출처럼 None이면 기본 namespace에서 검색한다.

        TODO(담당 A):
            - settings.embedding_model 로 query 임베딩 생성
            - pgvector `<->` 거리 연산자로 ORDER BY
            - topic 있으면 WHERE 절로 필터
            - EvidenceChunk 복원
        """

        namespace = self._namespace(user_id)
        chunks = self._chunks_by_user.get(namespace, [])

        # 현재 구현: 유사도 검색 대신 topic 일치 필터 + 앞에서부터 k개 반환.
        if topic is not None:
            chunks = [chunk for chunk in chunks if chunk.topic == topic]

        return chunks[:k]

    def build_coverage_map(self, user_id: int | str | None = None) -> CoverageMap:
        """저장된 청크들의 주제별 근거 커버리지를 집계한다.

        각 EvidenceChunk 를 topic 기준으로 묶고, 주제별 confidence 평균과
        청크 수를 계산한다. Strategy 는 이 결과를 보고 질문 주제를 고른다.

        Args:
            user_id: 사용자별 저장 namespace를 선택하기 위한 사용자 ID.
                기존 호출처럼 None이면 기본 namespace 기준으로 집계한다.

        Returns:
            주제별 평균 신뢰도와 근거 청크 수를 담은 CoverageMap.
        """

        namespace = self._namespace(user_id)
        chunks = self._chunks_by_user.get(namespace, [])
        # 현재 구현: topic 별 confidence 단순 평균.
        by_topic: dict[str, list[float]] = {}
        for c in chunks:
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

    def _namespace(self, user_id: int | str | None) -> str:
        """user_id를 stub 저장소에서 사용할 namespace 문자열로 변환한다."""
        return str(user_id) if user_id is not None else self.DEFAULT_NAMESPACE


# 런타임 공용 단일 인스턴스. retrieval.py 가 이걸 연다.
_store: EvidenceStore | None = None


def get_store() -> EvidenceStore:
    """런타임 공용 EvidenceStore 싱글톤을 반환한다.

    Retrieval Tool과 indexing 파이프라인이 같은 저장소 경계를 사용하도록 한다.
    현재는 프로세스 메모리에 유지되고, 실제 DB 구현 이후에도 호출부는 이
    함수로 store 인스턴스를 얻는다.
    """
    global _store
    if _store is None:
        _store = EvidenceStore()
    return _store
