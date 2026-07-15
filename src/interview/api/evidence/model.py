"""사용자가 마이페이지에서 등록한 Evidence 출처 링크 모델."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from interview.api.database import Base
from interview.config import settings


class EvidenceSourceLink(Base):
    """사용자별 Notion 또는 GitHub 등록 링크를 보관한다."""

    __tablename__ = "evidence_source_links"
    __table_args__ = (
        UniqueConstraint("user_id", "normalized_url", name="uq_evidence_source_links_user_url"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    normalized_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class EvidenceVectorRecord(Base):
    """사용자 Evidence 청크와 임베딩을 pgvector에 영속화한다."""

    __tablename__ = "evidence_vector_records"
    __table_args__ = (
        UniqueConstraint("user_id", "chunk_id", name="uq_evidence_vector_records_user_chunk"),
        Index("ix_evidence_vector_records_user_topic", "user_id", "topic"),
        Index(
            "ix_evidence_vector_records_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    chunk_id: Mapped[str] = mapped_column(String(128), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    topic: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    doc_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    week: Mapped[int | None] = mapped_column(Integer, nullable=True)
    date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    language: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ownership: Mapped[str | None] = mapped_column(String(32), nullable=True)
    commit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_commit_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimensions),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
