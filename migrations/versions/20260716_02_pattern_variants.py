"""add preserved pattern variants"""

from alembic import op
import sqlalchemy as sa

revision = "20260716_02"
down_revision = "20260716_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_question_patterns",
        sa.Column("variants", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.alter_column("interview_question_patterns", "variants", server_default=None)


def downgrade() -> None:
    op.drop_column("interview_question_patterns", "variants")
