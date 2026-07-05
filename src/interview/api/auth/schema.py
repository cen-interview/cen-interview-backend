"""로그인 요청/토큰 응답 스키마."""

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