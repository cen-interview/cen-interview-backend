"""전역 면접 질문 패턴의 적재 및 Strategy용 검색 경계."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from langchain_openai import OpenAIEmbeddings
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from interview.api.database import SessionLocal
from interview.api.evidence.question_pattern_model import InterviewQuestionPatternRecord
from interview.config import settings
from interview.schemas.question_pattern import InterviewQuestionSignal


class QuestionPatternStore:
    """PostgreSQL pgvector 기반 전역 패턴 저장소."""

    def __init__(
        self,
        *,
        embedding_client: Any | None = None,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self.embedding_client = embedding_client
        self.session_factory = session_factory

    def upsert(self, rows: list[dict[str, Any]], batch_size: int = 500) -> int:
        """최종 패턴 rows를 pattern_id 기준으로 대량 upsert한다."""
        if not rows:
            return 0
        with self._session() as db:
            try:
                for start in range(0, len(rows), max(1, batch_size)):
                    statement = pg_insert(InterviewQuestionPatternRecord).values(
                        rows[start : start + batch_size]
                    )
                    statement = statement.on_conflict_do_update(
                        constraint="uq_interview_question_patterns_pattern_id",
                        set_={
                            column: getattr(statement.excluded, column)
                            for column in (
                                "pattern_text",
                                "variants",
                                "frequency",
                                "signal_kind",
                                "required_evidence_signals",
                                "topic_family",
                                "embedding",
                                "dataset_version",
                                "updated_at",
                            )
                        },
                    )
                    db.execute(statement)
                db.commit()
            except Exception:
                db.rollback()
                raise
        return len(rows)

    def replace_dataset(
        self,
        rows: list[dict[str, Any]],
        dataset_version: str,
        batch_size: int = 500,
    ) -> int:
        """새 데이터셋으로 교체하고 이전 버전의 잔여 패턴을 삭제한다."""
        pattern_ids = {row["pattern_id"] for row in rows}
        with self._session() as db:
            try:
                if rows:
                    for start in range(0, len(rows), max(1, batch_size)):
                        statement = pg_insert(InterviewQuestionPatternRecord).values(
                            rows[start : start + batch_size]
                        )
                        db.execute(
                            statement.on_conflict_do_update(
                                constraint="uq_interview_question_patterns_pattern_id",
                                set_={
                                    column: getattr(statement.excluded, column)
                                    for column in (
                                        "pattern_text",
                                        "variants",
                                        "frequency",
                                        "signal_kind",
                                        "required_evidence_signals",
                                        "topic_family",
                                        "embedding",
                                        "dataset_version",
                                        "updated_at",
                                    )
                                },
                            )
                        )
                if pattern_ids:
                    db.execute(
                        delete(InterviewQuestionPatternRecord).where(
                            InterviewQuestionPatternRecord.pattern_id.not_in(pattern_ids)
                        )
                    )
                else:
                    db.execute(delete(InterviewQuestionPatternRecord))
                stored_count = db.scalar(
                    select(func.count(InterviewQuestionPatternRecord.id))
                )
                if stored_count != len(pattern_ids):
                    raise RuntimeError(
                        f"패턴 교체 후 행 수가 입력과 다릅니다: expected={len(pattern_ids)}, actual={stored_count}"
                    )
                db.commit()
            except Exception:
                db.rollback()
                raise
        return len(rows)

    def search(
        self,
        query: str,
        *,
        kind: str | None = None,
        topic_family: str | None = None,
        limit: int = 5,
        min_similarity: float | None = settings.question_pattern_min_similarity,
    ) -> list[InterviewQuestionSignal]:
        """질문 패턴을 cosine similarity 순서로 조회한다."""
        if not query.strip() or limit <= 0:
            return []
        if min_similarity is not None and not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity는 0과 1 사이여야 합니다.")
        embedding = self._embeddings().embed_query(query)
        distance = InterviewQuestionPatternRecord.embedding.cosine_distance(embedding)
        filters = []
        if kind is not None:
            filters.append(InterviewQuestionPatternRecord.signal_kind == kind)
        if topic_family is not None:
            filters.append(InterviewQuestionPatternRecord.topic_family == topic_family)
        if min_similarity is not None:
            filters.append(distance <= 1.0 - min_similarity)
        with self._session() as db:
            records = db.execute(
                select(InterviewQuestionPatternRecord, distance)
                .where(*filters)
                .order_by(distance)
                .limit(limit)
            ).all()
        return [
            InterviewQuestionSignal(
                pattern_id=record.pattern_id,
                pattern_text=record.pattern_text,
                frequency=record.frequency,
                signal_kind=record.signal_kind,
                required_evidence_signals=record.required_evidence_signals or [],
                topic_family=record.topic_family,
                similarity=max(0.0, min(1.0, 1.0 - float(record_distance))),
            )
            for record, record_distance in records
        ]

    def _embeddings(self) -> Any:
        if self.embedding_client is None:
            self.embedding_client = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
                dimensions=settings.embedding_dimensions,
            )
        return self.embedding_client

    @contextmanager
    def _session(self) -> Iterator[Session]:
        db = self.session_factory()
        try:
            yield db
        finally:
            db.close()


_pattern_store: QuestionPatternStore | None = None


def get_question_pattern_store() -> QuestionPatternStore:
    global _pattern_store
    if _pattern_store is None:
        _pattern_store = QuestionPatternStore()
    return _pattern_store


def search_interview_question_signals(
    query: str,
    *,
    kind: str | None = None,
    topic_family: str | None = None,
    limit: int = 5,
    min_similarity: float | None = settings.question_pattern_min_similarity,
) -> list[InterviewQuestionSignal]:
    """Strategy 전용 질문 패턴 신호 조회 함수.

    ``search_evidence``와 분리되어 Assessment의 사용자 근거 검색에는 영향을
    주지 않는다.
    """
    return get_question_pattern_store().search(
        query,
        kind=kind,
        topic_family=topic_family,
        limit=limit,
        min_similarity=min_similarity,
    )
