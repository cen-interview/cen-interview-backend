"""evidence_store: Evidence 청크 저장소 경계.

최종 목표는 청크를 임베딩해 Vector DB 에 적재하고, 쿼리로 유사 청크를
검색하는 것이다. 기본 구현은 테스트와 로컬 개발을 위한 user_id namespace별
인메모리 저장소이고, 설정으로 Chroma 기반 Vector DB 저장소를 켤 수 있다.

  - 적재(add_chunks): 인덱싱 파이프라인이 면접 전 1회 호출
  - 검색(query): Retrieval Tool(retrieval.py)이 런타임에 호출
"""

from collections.abc import Sequence
from typing import Any

from langchain_openai import OpenAIEmbeddings

from interview.config import settings
from interview.schemas.evidence import CoverageMap, EvidenceChunk, TopicCoverage

DEFAULT_TOP_K = 5
VECTOR_BACKEND_CHROMA = "chroma"
VECTOR_BACKEND_MEMORY = "memory"
_CHROMA_COLLECTION_NAME = "evidence_chunks"

class EvidenceStore:
    """Evidence 청크 저장/검색을 담당하는 저장소 경계.

    현재는 Postgres + pgvector 전환 전 임시 구현으로 user_id namespace별
    인메모리 저장소를 사용한다. 호출부는 이 클래스의 메서드 계약만 의존하게
    해서 실제 DB 구현으로 바뀌어도 Strategy/Assessment 영향을 줄인다.
    """

    DEFAULT_NAMESPACE = "default"

    def __init__(
        self,
        database_url: str | None = None,
        backend: str | None = None,
        embedding_client: Any | None = None,
    ) -> None:
        """EvidenceStore를 초기화하고 설정된 저장 backend를 준비한다.

        Args:
            database_url: pgvector 저장소로 전환할 때 사용할 DB URL.
                None이면 전역 settings.database_url을 사용한다.
            backend: ``memory`` 또는 ``chroma``. None이면 설정값을 사용한다.
            embedding_client: 테스트에서 주입할 embedding client. None이면
                OpenAIEmbeddings를 사용한다.
        """
        self.database_url = database_url or settings.database_url
        self.embedding_model = settings.embedding_model
        self.embedding_dimensions = settings.embedding_dimensions
        self.backend = (backend or settings.evidence_store_backend).lower()
        self.embedding_client = embedding_client
        self._conn = None
        self._collection = None

        # 기본 구현: 외부 서비스 없이 사용자별 인메모리 리스트에 청크를 보관한다.
        self._chunks_by_user: dict[str, list[EvidenceChunk]] = {}

    def add_chunks(
        self,
        chunks: list[EvidenceChunk],
        user_id: int | str | None = None,
    ) -> None:
        """청크를 저장한다.

        memory backend는 임베딩 없이 메모리에 보관한다. chroma backend는 여기서
        chunk.text 임베딩을 만들고 메타데이터와 함께 Vector DB에 upsert한다.

        Args:
            chunks: 저장할 EvidenceChunk 목록.
            user_id: 사용자별 저장 namespace를 선택하기 위한 사용자 ID.
                기존 호출처럼 None이면 기본 namespace에 저장한다.

        """
        namespace = self._namespace(user_id)

        if self.backend == VECTOR_BACKEND_CHROMA:
            self._add_chunks_to_chroma(chunks, namespace)
            return

        self._chunks_by_user.setdefault(namespace, []).extend(chunks)

    def clear_user(self, user_id: int | str | None = None) -> None:
        """사용자 namespace에 저장된 기존 청크를 삭제한다.

        같은 사용자가 Notion/GitHub 링크를 다시 등록하면 이전 인덱싱 결과가
        새 결과와 섞이지 않아야 한다. 실제 DB 구현에서는 이 메서드가
        user_id 조건 delete 또는 collection drop 역할을 한다.
        """
        namespace = self._namespace(user_id)

        if self.backend == VECTOR_BACKEND_CHROMA:
            collection = self._get_chroma_collection()
            try:
                collection.delete(where={"user_id": namespace})
            except ValueError:
                pass
            return

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

        chroma backend는 query 임베딩을 만들고 Vector DB에서 top-k를 검색한다.
        """

        namespace = self._namespace(user_id)
        if self.backend == VECTOR_BACKEND_CHROMA:
            return self._query_chroma(query=query, topic=topic, k=k, namespace=namespace)

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
        if self.backend == VECTOR_BACKEND_CHROMA:
            chunks = self._get_chroma_chunks(namespace)
            return self._coverage_from_chunks(chunks)

        chunks = self._chunks_by_user.get(namespace, [])

        return self._coverage_from_chunks(chunks)

    def _coverage_from_chunks(self, chunks: Sequence[EvidenceChunk]) -> CoverageMap:
        """청크 목록에서 topic 별 confidence 평균과 개수를 계산한다."""
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

    def _get_embeddings(self) -> Any:
        """설정된 embedding client를 반환한다."""
        if self.embedding_client is None:
            self.embedding_client = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=settings.openai_api_key,
                dimensions=self.embedding_dimensions,
            )
        return self.embedding_client

    def _get_chroma_collection(self) -> Any:
        """Chroma collection을 lazy 초기화한다."""
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=settings.evidence_chroma_path)
            self._collection = client.get_or_create_collection(_CHROMA_COLLECTION_NAME)
        return self._collection

    def _add_chunks_to_chroma(
        self,
        chunks: list[EvidenceChunk],
        namespace: str,
    ) -> None:
        """청크를 임베딩해 Chroma Vector DB에 저장한다."""
        if not chunks:
            return

        embeddings = self._get_embeddings().embed_documents([chunk.text for chunk in chunks])
        collection = self._get_chroma_collection()
        ids = [self._stored_chunk_id(namespace, chunk.chunk_id) for chunk in chunks]

        collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[self._metadata_from_chunk(chunk, namespace) for chunk in chunks],
        )

    def _query_chroma(
        self,
        *,
        query: str,
        topic: str | None,
        k: int,
        namespace: str,
    ) -> list[EvidenceChunk]:
        """Chroma Vector DB에서 사용자 namespace와 선택 topic 기준으로 검색한다."""
        collection = self._get_chroma_collection()
        query_embedding = self._get_embeddings().embed_query(query)
        where: dict[str, Any] = {"user_id": namespace}
        if topic is not None:
            where = {"$and": [where, {"topic": topic}]}

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            self._chunk_from_metadata(document=document, metadata=metadata)
            for document, metadata in zip(documents, metadatas, strict=False)
            if metadata is not None
        ]

    def _get_chroma_chunks(self, namespace: str) -> list[EvidenceChunk]:
        """Chroma에 저장된 특정 사용자 namespace의 모든 청크를 복원한다."""
        collection = self._get_chroma_collection()
        result = collection.get(
            where={"user_id": namespace},
            include=["documents", "metadatas"],
        )
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        return [
            self._chunk_from_metadata(document=document, metadata=metadata)
            for document, metadata in zip(documents, metadatas, strict=False)
            if metadata is not None
        ]

    def _metadata_from_chunk(self, chunk: EvidenceChunk, namespace: str) -> dict[str, Any]:
        """EvidenceChunk를 Vector DB metadata로 변환한다."""
        return {
            "chunk_id": chunk.chunk_id,
            "user_id": namespace,
            "source_type": str(chunk.source_type.value),
            "source_url": chunk.source_url,
            "topic": chunk.topic,
            "doc_type": chunk.doc_type or "",
            "week": chunk.week if chunk.week is not None else -1,
            "date": chunk.date or "",
            "confidence": chunk.confidence,
        }

    def _chunk_from_metadata(
        self,
        *,
        document: str,
        metadata: dict[str, Any],
    ) -> EvidenceChunk:
        """Vector DB metadata와 document에서 EvidenceChunk를 복원한다."""
        week = metadata.get("week")
        return EvidenceChunk(
            chunk_id=str(metadata["chunk_id"]),
            text=document,
            source_type=str(metadata["source_type"]),
            source_url=str(metadata["source_url"]),
            topic=str(metadata["topic"]),
            doc_type=str(metadata.get("doc_type") or "") or None,
            week=int(week) if week is not None and int(week) >= 0 else None,
            date=str(metadata.get("date") or "") or None,
            confidence=float(metadata["confidence"]),
        )

    def _stored_chunk_id(self, namespace: str, chunk_id: str) -> str:
        """Vector DB에서 사용자별로 충돌하지 않는 저장 ID를 만든다."""
        return f"{namespace}:{chunk_id}"


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
