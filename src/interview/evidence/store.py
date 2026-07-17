"""Evidence 청크의 pgvector 저장과 검색을 담당하는 저장소 경계."""

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from interview.api.database import SessionLocal
from interview.api.evidence.model import EvidenceVectorRecord
from interview.config import settings
from interview.schemas.evidence import (
    CoverageMap,
    EvidenceChunk,
    EvidenceOwnership,
    TopicCoverage,
)
import math
import re
from difflib import SequenceMatcher

DEFAULT_TOP_K = 5
VECTOR_BACKEND_MEMORY = "memory"
VECTOR_BACKEND_PGVECTOR = "pgvector"


class EvidenceStore:
    """사용자 Evidence를 pgvector에 저장하고 유사도 검색한다.

    호출부는 이 클래스의 ``add_chunks``/``query`` 계약만 사용한다. 운영에서는
    pgvector가 기본 backend이고, memory backend는 외부 DB 없이 실행하는 단위
    테스트에만 사용한다.
    """

    DEFAULT_NAMESPACE = "default"

    def __init__(
        self,
        database_url: str | None = None,
        backend: str | None = None,
        embedding_client: Any | None = None,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        """저장 backend, 임베딩 클라이언트, DB 세션 생성기를 준비한다."""
        self.database_url = database_url or settings.database_url
        self.embedding_model = settings.embedding_model
        self.embedding_dimensions = settings.embedding_dimensions
        self.backend = (backend or settings.evidence_store_backend).lower()
        if self.backend not in {VECTOR_BACKEND_MEMORY, VECTOR_BACKEND_PGVECTOR}:
            raise ValueError("EVIDENCE_STORE_BACKEND는 memory 또는 pgvector여야 합니다.")

        self.embedding_client = embedding_client
        self._chunks_by_user: dict[str, list[EvidenceChunk]] = {}
        if session_factory is not None:
            self._session_factory = session_factory
        elif database_url is not None and database_url != settings.database_url:
            engine = create_engine(database_url)
            self._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        else:
            self._session_factory = SessionLocal

    def add_chunks(
        self,
        chunks: list[EvidenceChunk],
        user_id: int | str | None = None,
    ) -> None:
        """청크를 임베딩해 사용자 namespace별로 upsert한다."""
        if not chunks:
            return

        namespace = self._namespace(user_id)
        if self.backend == VECTOR_BACKEND_MEMORY:
            self._chunks_by_user.setdefault(namespace, []).extend(chunks)
            return

        embeddings = self._embed_chunk_texts(chunks)
        if len(embeddings) != len(chunks):
            raise ValueError("임베딩 개수와 EvidenceChunk 개수가 일치하지 않습니다.")

        values = [
            self._record_values(chunk=chunk, namespace=namespace, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        with self._session() as db:
            try:
                for batch in _batches(values, settings.evidence_db_batch_size):
                    db.execute(self._upsert_statement(batch))
                db.commit()
            except Exception:
                db.rollback()
                raise

    def _embed_chunk_texts(self, chunks: Sequence[EvidenceChunk]) -> list[list[float]]:
        """중복 텍스트를 제거하고 제한된 배치 병렬 처리로 임베딩한다."""
        unique_texts = list(dict.fromkeys(chunk.text for chunk in chunks))
        batches = list(_batches(unique_texts, settings.evidence_embedding_batch_size))
        embeddings = self._get_embeddings()
        max_workers = max(
            1,
            min(settings.evidence_embedding_concurrency, len(batches)),
        )
        if max_workers == 1:
            embedded_batches = [embeddings.embed_documents(batch) for batch in batches]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                embedded_batches = list(executor.map(embeddings.embed_documents, batches))

        embedding_by_text = {
            text: embedding
            for text, embedding in zip(
                unique_texts,
                [item for batch in embedded_batches for item in batch],
                strict=True,
            )
        }
        return [embedding_by_text[chunk.text] for chunk in chunks]

    def _upsert_statement(self, values: list[dict[str, Any]]) -> Any:
        """Evidence 레코드 한 배치에 대한 PostgreSQL upsert 문을 만든다."""
        statement = pg_insert(EvidenceVectorRecord).values(values)
        return statement.on_conflict_do_update(
            constraint="uq_evidence_vector_records_user_chunk",
            set_={
                column: getattr(statement.excluded, column)
                for column in (
                    "text",
                    "source_type",
                    "source_url",
                    "topic",
                    "doc_type",
                    "week",
                    "date",
                    "confidence",
                    "file_path",
                    "language",
                    "ownership",
                    "commit_count",
                    "last_commit_sha",
                    "embedding",
                    "updated_at",
                )
            },
        )

    def clear_user(self, user_id: int | str | None = None) -> None:
        """사용자의 이전 Evidence 전체를 삭제한다."""
        namespace = self._namespace(user_id)
        if self.backend == VECTOR_BACKEND_MEMORY:
            self._chunks_by_user[namespace] = []
            return
        self._delete_records(EvidenceVectorRecord.user_id == namespace)

    def clear_user_sources(
        self,
        source_types: set[str],
        user_id: int | str | None = None,
    ) -> None:
        """부분 재인덱싱 대상 source의 청크만 삭제한다."""
        if not source_types:
            return
        namespace = self._namespace(user_id)
        if self.backend == VECTOR_BACKEND_MEMORY:
            self._chunks_by_user[namespace] = [
                chunk
                for chunk in self._chunks_by_user.get(namespace, [])
                if chunk.source_type.value not in source_types
            ]
            return
        self._delete_records(
            EvidenceVectorRecord.user_id == namespace,
            EvidenceVectorRecord.source_type.in_(source_types),
        )

    def query(
        self,
        query: str,
        topic: str | None = None,
        k: int = DEFAULT_TOP_K,
        user_id: int | str | None = None,
        ownership: EvidenceOwnership | None = None,
    ) -> list[EvidenceChunk]:
        """사용자 namespace에서 최소 cosine similarity를 충족하는 청크를 거리 순으로 최대 k개 반환한다."""
        namespace = self._namespace(user_id)
        if self.backend == VECTOR_BACKEND_MEMORY:
            chunks = self._chunks_by_user.get(namespace, [])
            if topic is not None:
                chunks = [chunk for chunk in chunks if chunk.topic == topic]
            if ownership is not None:
                chunks = [chunk for chunk in chunks if chunk.ownership == ownership]
            return chunks[:k]

        query_embedding = self._get_embeddings().embed_query(query)
        distance = EvidenceVectorRecord.embedding.cosine_distance(query_embedding)
        max_distance = 1.0 - settings.evidence_min_similarity

        statement = select(EvidenceVectorRecord).where(
            EvidenceVectorRecord.user_id == namespace
        )
        if topic is not None:
            statement = statement.where(EvidenceVectorRecord.topic == topic)
        if ownership is not None:
            statement = statement.where(EvidenceVectorRecord.ownership == ownership)

        statement = statement.where(distance <= max_distance).order_by(distance).limit(k)

        with self._session() as db:
            records = db.scalars(statement).all()
        return [self._chunk_from_record(record) for record in records]
    
     # 우지연 추가
    def _deduplicate_chunks(
        self,
        chunks: list[EvidenceChunk],
    ) -> list[EvidenceChunk]:
        """동일하거나 거의 같은 Evidence를 제거한다."""

        selected: list[EvidenceChunk] = []
        normalized_texts: list[str] = []

        for chunk in chunks:
            normalized = re.sub(
                r"\s+",
                " ",
                chunk.text,
            ).strip().lower()

            is_duplicate = any(
                SequenceMatcher(
                    None,
                    normalized,
                    existing,
                ).ratio() >= 0.95
                for existing in normalized_texts
            )

            if is_duplicate:
                continue

            selected.append(chunk)
            normalized_texts.append(normalized)

        return selected


    # 우지연 추가
    def rank_chunks_by_query(
        self,
        query_text: str,
        chunk_ids: list[str],
        k: int = 3,
        user_id: int | str | None = None,
    ) -> list[EvidenceChunk]:
        """지정된 Evidence 중 질문과 유사한 chunk를 최대 k개 반환한다."""

        if not query_text.strip() or not chunk_ids or k <= 0:
            return []

        namespace = self._namespace(user_id)
        unique_chunk_ids = list(dict.fromkeys(chunk_ids))

        # 테스트용 memory backend
        if self.backend == VECTOR_BACKEND_MEMORY:
            chunks = self.get_chunks_by_ids(
                chunk_ids=unique_chunk_ids,
                user_id=user_id,
            )

            if not chunks:
                return []

            embeddings = self._get_embeddings()
            query_embedding = embeddings.embed_query(query_text)
            chunk_embeddings = embeddings.embed_documents(
                [chunk.text for chunk in chunks]
            )

            def cosine_similarity(
                left: list[float],
                right: list[float],
            ) -> float:
                dot_product = sum(
                    left_value * right_value
                    for left_value, right_value in zip(
                        left,
                        right,
                        strict=True,
                    )
                )
                left_norm = math.sqrt(
                    sum(value * value for value in left)
                )
                right_norm = math.sqrt(
                    sum(value * value for value in right)
                )

                if left_norm == 0.0 or right_norm == 0.0:
                    return 0.0

                return dot_product / (left_norm * right_norm)

            scored_chunks = [
                (
                    cosine_similarity(
                        query_embedding,
                        chunk_embedding,
                    ),
                    chunk,
                )
                for chunk, chunk_embedding in zip(
                    chunks,
                    chunk_embeddings,
                    strict=True,
                )
            ]

            scored_chunks.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            ranked_chunks = [
                chunk
                for _, chunk in scored_chunks[:k]
            ]

            return self._deduplicate_chunks(ranked_chunks)

        # 운영 pgvector backend
        query_embedding = self._get_embeddings().embed_query(
            query_text
        )

        distance = EvidenceVectorRecord.embedding.cosine_distance(
            query_embedding
        )

        statement = (
            select(EvidenceVectorRecord)
            .where(
                EvidenceVectorRecord.user_id == namespace,
                EvidenceVectorRecord.chunk_id.in_(
                    unique_chunk_ids
                ),
            )
            .order_by(distance)
            .limit(k)
        )

        with self._session() as db:
            records = db.scalars(statement).all()

        ranked_chunks = [
            self._chunk_from_record(record)
            for record in records
        ]

        return self._deduplicate_chunks(ranked_chunks)
    
    # 우지연추가 
    def get_chunks_by_ids(
        self,
        chunk_ids: list[str],
        user_id: int | str | None = None,
        ) -> list[EvidenceChunk]:
        """지정된 ID에 해당하는 Evidence 원문을 조회한다."""

        if not chunk_ids:
            return []

        namespace = self._namespace(user_id)
        unique_chunk_ids = list(dict.fromkeys(chunk_ids))

        if self.backend == VECTOR_BACKEND_MEMORY:
            chunks = self._chunks_by_user.get(namespace, [])

            chunk_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
            }

            return [
                chunk_by_id[chunk_id]
                for chunk_id in unique_chunk_ids
                if chunk_id in chunk_by_id
            ]

        statement = select(EvidenceVectorRecord).where(
            EvidenceVectorRecord.user_id == namespace,
            EvidenceVectorRecord.chunk_id.in_(unique_chunk_ids),
        )

        with self._session() as db:
            records = db.scalars(statement).all()

        record_by_id = {
            record.chunk_id: record
            for record in records
        }

        return [
            self._chunk_from_record(record_by_id[chunk_id])
            for chunk_id in unique_chunk_ids
            if chunk_id in record_by_id
        ]   
    
    def build_coverage_map(self, user_id: int | str | None = None) -> CoverageMap:
        """사용자별 저장 청크의 topic confidence 평균과 개수를 집계한다."""
        namespace = self._namespace(user_id)
        if self.backend == VECTOR_BACKEND_MEMORY:
            return self._coverage_from_chunks(self._chunks_by_user.get(namespace, []))

        statement = (
            select(
                EvidenceVectorRecord.topic,
                func.avg(EvidenceVectorRecord.confidence),
                func.count(EvidenceVectorRecord.id),
            )
            .where(EvidenceVectorRecord.user_id == namespace)
            .group_by(EvidenceVectorRecord.topic)
        )
        with self._session() as db:
            rows = db.execute(statement).all()
        return CoverageMap(
            topic_coverage={
                str(topic): TopicCoverage(
                    confidence=float(confidence),
                    chunk_count=int(chunk_count),
                )
                for topic, confidence, chunk_count in rows
            },
            updated_at=None,
        )

    def _delete_records(self, *conditions: Any) -> None:
        """지정한 조건의 pgvector 레코드를 트랜잭션으로 삭제한다."""
        with self._session() as db:
            try:
                db.execute(delete(EvidenceVectorRecord).where(*conditions))
                db.commit()
            except Exception:
                db.rollback()
                raise

    def _coverage_from_chunks(self, chunks: Sequence[EvidenceChunk]) -> CoverageMap:
        """memory backend 테스트용 청크 목록을 CoverageMap으로 집계한다."""
        by_topic: dict[str, list[float]] = {}
        for chunk in chunks:
            by_topic.setdefault(chunk.topic, []).append(chunk.confidence)
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

    def _get_embeddings(self) -> Any:
        """설정된 OpenAI 임베딩 모델을 lazy 초기화한다."""
        if self.embedding_client is None:
            self.embedding_client = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=settings.openai_api_key,
                dimensions=self.embedding_dimensions,
            )
        return self.embedding_client

    def _record_values(
        self,
        *,
        chunk: EvidenceChunk,
        namespace: str,
        embedding: list[float],
    ) -> dict[str, Any]:
        """EvidenceChunk와 임베딩을 PostgreSQL upsert 입력으로 변환한다."""
        return {
            "user_id": namespace,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source_type": chunk.source_type.value,
            "source_url": chunk.source_url,
            "topic": chunk.topic,
            "doc_type": chunk.doc_type,
            "week": chunk.week,
            "date": chunk.date,
            "confidence": chunk.confidence,
            "file_path": chunk.file_path,
            "language": chunk.language,
            "ownership": chunk.ownership,
            "commit_count": chunk.commit_count,
            "last_commit_sha": chunk.last_commit_sha,
            "embedding": embedding,
            "updated_at": datetime.utcnow(),
        }

    def _chunk_from_record(self, record: EvidenceVectorRecord) -> EvidenceChunk:
        """pgvector ORM 레코드를 런타임 공용 EvidenceChunk로 복원한다."""
        return EvidenceChunk(
            chunk_id=record.chunk_id,
            text=record.text,
            source_type=record.source_type,
            source_url=record.source_url,
            topic=record.topic,
            doc_type=record.doc_type,
            week=record.week,
            date=record.date,
            confidence=record.confidence,
            file_path=record.file_path,
            language=record.language,
            ownership=record.ownership,
            commit_count=record.commit_count,
            last_commit_sha=record.last_commit_sha,
        )

    def _namespace(self, user_id: int | str | None) -> str:
        """user_id를 저장·검색에 사용할 문자열 namespace로 정규화한다."""
        return str(user_id) if user_id is not None else self.DEFAULT_NAMESPACE

    @contextmanager
    def _session(self) -> Iterator[Session]:
        """저장소 내부 DB 세션의 close를 보장한다."""
        db = self._session_factory()
        try:
            yield db
        finally:
            db.close()


_store: EvidenceStore | None = None


def get_store() -> EvidenceStore:
    """인덱싱과 런타임 검색이 공유하는 EvidenceStore 싱글톤을 반환한다."""
    global _store
    if _store is None:
        _store = EvidenceStore()
    return _store


def _batches(items: Sequence[Any], batch_size: int) -> Iterator[list[Any]]:
    """입력 순서를 유지하며 설정된 크기의 리스트 배치를 만든다."""
    size = max(1, batch_size)
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
