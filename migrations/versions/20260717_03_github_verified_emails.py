"""store verified GitHub commit emails"""

from alembic import op
import sqlalchemy as sa


revision = "20260717_03"
down_revision = "20260716_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "github_credentials",
        sa.Column(
            "verified_emails",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )
    op.alter_column("github_credentials", "verified_emails", server_default=None)


def downgrade() -> None:
    op.drop_column("github_credentials", "verified_emails")
