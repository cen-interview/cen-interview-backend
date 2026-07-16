"""사용자 동의 후 공개되는 기술 질문 rubric DB 모델."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from interview.api.database import Base
from interview.config import settings


class RubricSetRecord(Base):
    """한 질문에 대한 rubric 전체의 검증 상태를 보관한다."""

    __tablename__ = "rubric_sets"
    __table_args__ = (
        UniqueConstraint(
            "question_id", "rubric_version",
            name="uq_rubric_sets_question_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    question_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimensions), nullable=True
    )
    rubric_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow,
        onupdate=datetime.utcnow, nullable=False,
    )


class RubricVectorRecord(Base):
    """기술 질문의 정답 요소와 임베딩을 criterion 단위로 보관한다."""

    __tablename__ = "rubric_vector_records"
    __table_args__ = (
        UniqueConstraint(
            "rubric_set_id", "criterion_id",
            name="uq_rubric_vector_records_set_criterion",
        ),
        Index("ix_rubric_vector_records_question", "question_id"),
        Index(
            "ix_rubric_vector_records_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    rubric_set_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("rubric_sets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[str] = mapped_column(String(128), nullable=False)
    topic: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    criterion_id: Mapped[str] = mapped_column(String(128), nullable=False)
    criterion_text: Mapped[str] = mapped_column(Text, nullable=False)
    required: Mapped[bool] = mapped_column(nullable=False, default=True)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    rubric_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimensions), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )


class RubricQuestionAlias(Base):
    """같은 rubric set으로 판단된 질문의 다양한 표현을 보관한다."""

    __tablename__ = "rubric_question_aliases"
    __table_args__ = (
        UniqueConstraint(
            "rubric_set_id",
            "source_question_id",
            name="uq_rubric_question_alias_source",
        ),
        Index("ix_rubric_question_alias_set", "rubric_set_id"),
        Index(
            "ix_rubric_question_alias_embedding_hnsw",
            "question_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"question_embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    rubric_set_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("rubric_sets.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_question_id: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimensions), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
