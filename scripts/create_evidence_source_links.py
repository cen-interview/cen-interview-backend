"""Evidence 링크 테이블을 현재 데이터베이스에 생성한다.

Alembic이 아직 도입되지 않은 개발 환경에서만 사용한다. 이미 존재하는 테이블은
변경하지 않는다.
"""

from interview.api.evidence.model import EvidenceSourceLink
from interview.api.database import engine
from interview.api.users.model import User


def main() -> None:
    """사용자별 등록 링크 테이블을 idempotent하게 생성한다."""

    _ = User  # ForeignKey("users.id") 해석을 위해 users 모델을 metadata에 등록한다.
    EvidenceSourceLink.__table__.create(bind=engine, checkfirst=True)
    print("evidence_source_links table is ready")


if __name__ == "__main__":
    main()
