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

class NotionCredential(Base):
    """사용자별 Notion OAuth credential을 저장하는 테이블 모델."""

    __tablename__ = "notion_credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
          Integer,
          ForeignKey("users.id"),
          nullable=False,
          unique=True,
          index=True,
    )

    mcp_access_token: Mapped[str] = mapped_column(String, nullable=False)
    mcp_refresh_token: Mapped[str | None] = mapped_column(String, nullable=True)
    mcp_client_id: Mapped[str] = mapped_column(String, nullable=False)
    mcp_client_secret: Mapped[str | None] = mapped_column(String, nullable=True)

    workspace_id: Mapped[str | None] = mapped_column(String, nullable=True)
    workspace_name: Mapped[str | None] = mapped_column(String, nullable=True)
    bot_id: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
          DateTime,
          default=datetime.utcnow,
          nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user = relationship("User")


class GitHubCredential(Base):
    """사용자별 GitHub OAuth credential을 저장하는 테이블 모델."""

    __tablename__ = "github_credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
          Integer,
          ForeignKey("users.id"),
          nullable=False,
          unique=True,
          index=True,
    )

    access_token: Mapped[str] = mapped_column(String, nullable=False)
    token_type: Mapped[str | None] = mapped_column(String, nullable=True)
    scope: Mapped[str | None] = mapped_column(String, nullable=True)

    github_user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    github_login: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
          DateTime,
          default=datetime.utcnow,
          nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user = relationship("User")
