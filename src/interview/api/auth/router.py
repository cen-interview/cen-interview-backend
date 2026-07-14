"""인증 관련 API 라우터."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from interview.api.core.security import decode_access_token
from interview.api.auth.dependency import get_current_user
from interview.api.auth.schema import LoginRequest, RefreshRequest, TokenResponse
from interview.api.auth.service import (
    build_github_oauth_authorize_url,
    build_notion_mcp_authorize_url,
    exchange_github_oauth_code,
    exchange_notion_mcp_code,
    login_user,
    logout_user,
    refresh_access_token as refresh_access_token_service,
    register_notion_mcp_client,
)
from interview.api.database import get_db
from interview.api.users.model import User

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """로그인 후 Access Token과 Refresh Token을 발급한다."""

    return login_user(db, req)


@router.post("/refresh")
def refresh_access_token(req: RefreshRequest, db: Session = Depends(get_db)):
    """Refresh Token으로 새로운 Access Token을 발급한다."""

    return refresh_access_token_service(db, req)

@router.get("/notion/start")
def start_notion_oauth(current_user: User = Depends(get_current_user)):
    """현재 로그인 사용자를 Notion MCP OAuth 승인 화면으로 보낸다."""

    client_data = register_notion_mcp_client()
    auth_data = build_notion_mcp_authorize_url(
        user=current_user,
        client_data=client_data,
    )

    return RedirectResponse(url=auth_data["authorize_url"])

@router.get("/notion/callback")
def notion_oauth_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Notion OAuth code를 access token으로 교환하고 사용자 credential로 저장한다."""

    if error is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Notion OAuth failed: {error}",
        )

    if code is None or state is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Notion OAuth callback에 code/state가 없습니다.",
        )

    payload = decode_access_token(state)
    if payload is None or payload.get("purpose") != "notion_mcp_oauth":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 Notion OAuth state입니다.",
        )

    return exchange_notion_mcp_code(
      db=db,
      user_id=payload["user_id"],
      code=code,
      client_id=payload["client_id"],
      client_secret=payload.get("client_secret"),
      code_verifier=payload["code_verifier"],
    )


@router.get("/github/start")
def start_github_oauth(current_user: User = Depends(get_current_user)):
    """현재 로그인 사용자를 GitHub OAuth 승인 화면으로 보낸다."""

    auth_data = build_github_oauth_authorize_url(user=current_user)
    return RedirectResponse(url=auth_data["authorize_url"])


@router.get("/github/callback")
def github_oauth_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """GitHub OAuth code를 access token으로 교환하고 사용자 credential로 저장한다."""

    if error is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"GitHub OAuth failed: {error}",
        )

    if code is None or state is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub OAuth callback에 code/state가 없습니다.",
        )

    payload = decode_access_token(state)
    if payload is None or payload.get("purpose") != "github_oauth":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 GitHub OAuth state입니다.",
        )

    return exchange_github_oauth_code(
        db=db,
        user_id=payload["user_id"],
        code=code,
    )


@router.post("/logout")
def logout(req: RefreshRequest, db: Session = Depends(get_db)):
    """Refresh Token을 삭제하여 로그아웃한다."""

    return logout_user(db, req)
