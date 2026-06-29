from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship

from interview.api.database import Base


class RefreshToken(Base):
    """Refresh Token 정보를 저장하는 테이블 모델."""

    __tablename__ = "refresh_tokens"

    # Refresh Token 테이블의 기본키
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # 실제 Refresh Token 문자열
    token: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)

    # 어떤 사용자의 Refresh Token인지 연결하기 위한 User ID
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
    )

    # Refresh Token 만료 시간
    expires_at = Column(DateTime(timezone=True), nullable=False)

    # 토큰 생성 시간
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # User 모델과 연결
    user = relationship("User")