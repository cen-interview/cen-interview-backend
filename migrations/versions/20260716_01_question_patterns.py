"""create global interview question pattern store"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "20260716_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "interview_question_patterns",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("pattern_id", sa.String(length=128), nullable=False),
        sa.Column("pattern_text", sa.Text(), nullable=False),
        sa.Column("frequency", sa.Integer(), nullable=False),
        sa.Column("signal_kind", sa.String(length=64), nullable=False),
        sa.Column("required_evidence_signals", sa.JSON(), nullable=False),
        sa.Column("topic_family", sa.String(length=128), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("dataset_version", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("pattern_id", name="uq_interview_question_patterns_pattern_id"),
    )
    op.create_index("ix_interview_question_patterns_signal_kind", "interview_question_patterns", ["signal_kind"])
    op.create_index("ix_interview_question_patterns_topic_family", "interview_question_patterns", ["topic_family"])
    op.create_index("ix_interview_question_patterns_dataset_version", "interview_question_patterns", ["dataset_version"])
    op.create_index(
        "ix_interview_question_patterns_embedding_hnsw",
        "interview_question_patterns",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_interview_question_patterns_embedding_hnsw", table_name="interview_question_patterns")
    op.drop_index("ix_interview_question_patterns_topic_family", table_name="interview_question_patterns")
    op.drop_index("ix_interview_question_patterns_dataset_version", table_name="interview_question_patterns")
    op.drop_index("ix_interview_question_patterns_signal_kind", table_name="interview_question_patterns")
    op.drop_table("interview_question_patterns")
