"""로그인 요청/토큰 응답 스키마."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    """로그인 요청 DTO."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """로그인 성공 시 반환할 Access Token + Refresh Token 응답 DTO."""

    # API 요청에 사용할 Access Token
    access_token: str

    # Access Token 재발급에 사용할 Refresh Token
    refresh_token: str

    # 토큰 타입
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    """Access Token 재발급 요청 DTO."""

    # 클라이언트가 가지고 있는 Refresh Token
    refresh_token: str


class OAuthAuthorizeUrlResponse(BaseModel):
    """프론트엔드가 브라우저 이동에 사용할 OAuth 승인 URL."""

    authorize_url: str


class OAuthConnectionStatus(BaseModel):
    """현재 서비스 사용자와 외부 OAuth 계정의 연결 상태."""

    provider: Literal["github", "notion"]
    connected: bool
    account_name: str | None = None
    account_id: str | None = None
    scope: str | None = None
    connected_at: datetime | None = None
