"""인증 관련 비즈니스 로직."""

import base64
import httpx
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from interview.config import settings
from urllib.parse import urlencode

from interview.api.auth.model import GitHubCredential, NotionCredential, RefreshToken
from interview.api.auth.schema import LoginRequest, RefreshRequest, TokenResponse
from interview.api.core.security import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    verify_password,
)
from interview.api.users.model import User

def login_user(db: Session, req: LoginRequest) -> TokenResponse:
    """사용자를 인증하고 Access Token과 Refresh Token을 발급한다."""

    user = db.query(User).filter(User.email == req.email).first()

    if not user or not verify_password(req.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다.",
        )

    token_data = {
        "sub": user.email,
        "user_id": user.id,
    }
    access_token = create_access_token(data=token_data)
    refresh_token = create_refresh_token(data=token_data)

    db_refresh_token = RefreshToken(
        token=refresh_token,
        user_id=user.id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )

    db.add(db_refresh_token)
    db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


def refresh_access_token(db: Session, req: RefreshRequest) -> dict[str, str]:
    """Refresh Token을 검증하고 새로운 Access Token을 발급한다."""

    payload = decode_access_token(req.refresh_token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 Refresh Token입니다.",
        )

    email = payload.get("sub")
    user_id = payload.get("user_id")

    if email is None or user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh Token 정보가 올바르지 않습니다.",
        )

    db_refresh_token = (
        db.query(RefreshToken)
        .filter(RefreshToken.token == req.refresh_token)
        .first()
    )

    if db_refresh_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="저장되지 않은 Refresh Token입니다.",
        )

    if db_refresh_token.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="만료된 Refresh Token입니다.",
        )

    user = db.query(User).filter(User.id == user_id, User.email == email).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자를 찾을 수 없습니다.",
        )

    return {
        "access_token": create_access_token(
            data={
                "sub": user.email,
                "user_id": user.id,
            }
        ),
        "token_type": "bearer",
    }

def _build_pkce_pair() -> tuple[str, str]:
      """OAuth PKCE용 code_verifier와 S256 code_challenge를 생성한다."""
      code_verifier = secrets.token_urlsafe(64)
      digest = hashlib.sha256(code_verifier.encode()).digest()
      code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
      return code_verifier, code_challenge

def register_notion_mcp_client() -> dict:
    """Notion MCP OAuth client를 동적으로 등록한다."""

    response = httpx.post(
        f"{settings.notion_mcp_issuer}/register",
        headers={
              "Accept": "application/json",
              "Content-Type": "application/json",
        },
        json={
              "client_name": "cen-interview-backend",
              "redirect_uris": [settings.notion_redirect_uri],
              "grant_types": ["authorization_code", "refresh_token"],
              "response_types": ["code"],
              "token_endpoint_auth_method": "none",
        },
        timeout=10.0,
    )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                  "message": "Notion MCP client registration에 실패했습니다.",
                  "notion_response": response.json(),
            },
        )

    client_data = response.json()

    if not client_data.get("client_id"):
          raise HTTPException(
              status_code=status.HTTP_400_BAD_REQUEST,
              detail="Notion MCP client registration 응답에 client_id가 없습니다.",
        )

    return client_data

def build_notion_mcp_authorize_url(
      *,
      user: User,
      client_data: dict,
  ) -> dict:
      """PKCE와 resource를 포함한 Notion MCP authorize URL을 만든다."""

      client_id = client_data.get("client_id")
      if not client_id:
          raise HTTPException(
              status_code=status.HTTP_400_BAD_REQUEST,
              detail="Notion MCP client_id가 없습니다.",
          )

      code_verifier, code_challenge = _build_pkce_pair()

      state = create_access_token(
          {
              "sub": user.email,
              "user_id": user.id,
              "purpose": "notion_mcp_oauth",
              "client_id": client_id,
              "client_secret": client_data.get("client_secret"),
              "code_verifier": code_verifier,
          }
      )

      query = urlencode(
          {
              "response_type": "code",
              "client_id": client_id,
              "redirect_uri": settings.notion_redirect_uri,
              "code_challenge": code_challenge,
              "code_challenge_method": "S256",
              "resource": settings.notion_mcp_resource,
              "state": state,
          }
      )

      return {
          "authorize_url": f"{settings.notion_mcp_issuer}/authorize?{query}",
          "state": state,
          "code_verifier": code_verifier,
          "client_id": client_id,
          "client_secret": client_data.get("client_secret"),
      }


def exchange_notion_mcp_code(
      *,
      db: Session,
      user_id: int,
      code: str,
      client_id: str,
      client_secret: str | None,
      code_verifier: str,
  ) -> dict:
      """Notion MCP authorization code를 MCP access token으로 교환하고 저장한다.

      Args:
          db: 요청 범위의 DB 세션.
          user_id: Notion MCP 연결을 요청한 서비스 사용자 ID.
          code: Notion MCP callback으로 전달된 authorization code.
          client_id: Dynamic client registration으로 발급받은 client_id.
          client_secret: Dynamic client registration 응답의 client_secret.
              token_endpoint_auth_method=none이면 없을 수 있다.
          code_verifier: authorize URL 생성 시 만든 PKCE verifier.

      Returns:
          Notion MCP 연결 결과 요약. token 원문은 응답하지 않는다.
      """
      token_payload = {
          "grant_type": "authorization_code",
          "code": code,
          "redirect_uri": settings.notion_redirect_uri,
          "client_id": client_id,
          "code_verifier": code_verifier,
          "resource": settings.notion_mcp_resource,
      }

      if client_secret:
          token_payload["client_secret"] = client_secret

      response = httpx.post(
        f"{settings.notion_mcp_issuer}/token",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data=token_payload,
        timeout=10.0,
      )

      if response.status_code >= 400:
          raise HTTPException(
              status_code=status.HTTP_400_BAD_REQUEST,
              detail={
                  "message": "Notion MCP token exchange에 실패했습니다.",
                  "notion_response": response.json(),
              },
          )

      token_data = response.json()
      access_token = token_data.get("access_token")

      if not access_token:
          raise HTTPException(
              status_code=status.HTTP_400_BAD_REQUEST,
              detail="Notion MCP token 응답에 access_token이 없습니다.",
          )

      credential = (
          db.query(NotionCredential)
          .filter(NotionCredential.user_id == user_id)
          .first()
      )

      if credential is None:
          credential = NotionCredential(user_id=user_id)
          db.add(credential)

      credential.mcp_access_token = access_token
      credential.mcp_refresh_token = token_data.get("refresh_token")
      credential.mcp_client_id = client_id
      credential.mcp_client_secret = client_secret
      credential.workspace_id = token_data.get("workspace_id")
      credential.workspace_name = token_data.get("workspace_name")
      credential.bot_id = token_data.get("bot_id")

      db.commit()
      db.refresh(credential)

      return {
          "connected": True,
          "credential_id": credential.id,
          "user_id": credential.user_id,
          "workspace_id": credential.workspace_id,
          "workspace_name": credential.workspace_name,
          "bot_id": credential.bot_id,
      }


def build_github_oauth_authorize_url(*, user: User) -> dict:
    """현재 사용자를 GitHub OAuth 승인 화면으로 보내기 위한 URL을 만든다."""

    if not settings.github_oauth_client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GitHub OAuth client_id가 설정되지 않았습니다.",
        )

    state = create_access_token(
        {
            "sub": user.email,
            "user_id": user.id,
            "purpose": "github_oauth",
        }
    )

    query = urlencode(
        {
            "client_id": settings.github_oauth_client_id,
            "redirect_uri": settings.github_oauth_redirect_uri,
            "scope": settings.github_oauth_scope,
            "state": state,
        }
    )

    return {
        "authorize_url": f"https://github.com/login/oauth/authorize?{query}",
        "state": state,
    }


def exchange_github_oauth_code(
    *,
    db: Session,
    user_id: int,
    code: str,
) -> dict:
    """GitHub OAuth code를 access token으로 교환하고 사용자 credential로 저장한다."""

    if not settings.github_oauth_client_id or not settings.github_oauth_client_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GitHub OAuth client 설정이 없습니다.",
        )

    token_response = httpx.post(
        "https://github.com/login/oauth/access_token",
        headers={
            "Accept": "application/json",
        },
        data={
            "client_id": settings.github_oauth_client_id,
            "client_secret": settings.github_oauth_client_secret,
            "code": code,
            "redirect_uri": settings.github_oauth_redirect_uri,
        },
        timeout=10.0,
    )

    if token_response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "GitHub OAuth token exchange에 실패했습니다.",
                "github_response": token_response.json(),
            },
        )

    token_data = token_response.json()
    access_token = token_data.get("access_token")

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "GitHub OAuth token 응답에 access_token이 없습니다.",
                "github_response": token_data,
            },
        )

    user_response = httpx.get(
        "https://api.github.com/user",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {access_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=10.0,
    )

    if user_response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "GitHub 사용자 정보 조회에 실패했습니다.",
                "github_response": user_response.json(),
            },
        )

    github_user = user_response.json()
    credential = (
        db.query(GitHubCredential)
        .filter(GitHubCredential.user_id == user_id)
        .first()
    )

    if credential is None:
        credential = GitHubCredential(user_id=user_id)
        db.add(credential)

    credential.access_token = access_token
    credential.token_type = token_data.get("token_type")
    credential.scope = token_data.get("scope")
    credential.github_user_id = str(github_user.get("id")) if github_user.get("id") else None
    credential.github_login = github_user.get("login")

    db.commit()
    db.refresh(credential)

    return {
        "connected": True,
        "credential_id": credential.id,
        "user_id": credential.user_id,
        "github_user_id": credential.github_user_id,
        "github_login": credential.github_login,
        "scope": credential.scope,
    }


def logout_user(db: Session, req: RefreshRequest) -> dict[str, str]:
    """Refresh Token을 삭제하여 로그아웃 처리한다."""

    db_refresh_token = (
        db.query(RefreshToken)
        .filter(RefreshToken.token == req.refresh_token)
        .first()
    )

    if db_refresh_token is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="이미 로그아웃되었거나 존재하지 않는 Refresh Token입니다.",
        )

    db.delete(db_refresh_token)
    db.commit()

    return {"message": "로그아웃이 완료되었습니다."}
