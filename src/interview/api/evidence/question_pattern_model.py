"""Strategy가 참조하는 전역 면접 질문 패턴 테이블."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from interview.api.database import Base
from interview.config import settings


class InterviewQuestionPatternRecord(Base):
    """사용자와 무관하게 배포되는 질문 패턴과 검색 임베딩."""

    __tablename__ = "interview_question_patterns"
    __table_args__ = (
        UniqueConstraint("pattern_id", name="uq_interview_question_patterns_pattern_id"),
        Index("ix_interview_question_patterns_signal_kind", "signal_kind"),
        Index("ix_interview_question_patterns_topic_family", "topic_family"),
        Index(
            "ix_interview_question_patterns_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pattern_id: Mapped[str] = mapped_column(String(128), nullable=False)
    pattern_text: Mapped[str] = mapped_column(Text, nullable=False)
    variants: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    frequency: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    signal_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    required_evidence_signals: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    topic_family: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimensions), nullable=False
    )
    dataset_version: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
