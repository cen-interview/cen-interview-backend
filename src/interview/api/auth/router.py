"""인증 관련 API 라우터."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from interview.api.auth.schema import LoginRequest, RefreshRequest, TokenResponse
from interview.api.auth.service import (
    login_user,
    logout_user,
    refresh_access_token as refresh_access_token_service,
)
from interview.api.database import get_db


router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """로그인 후 Access Token과 Refresh Token을 발급한다."""

    return login_user(db, req)


@router.post("/refresh")
def refresh_access_token(req: RefreshRequest, db: Session = Depends(get_db)):
    """Refresh Token으로 새로운 Access Token을 발급한다."""

    return refresh_access_token_service(db, req)


@router.post("/logout")
def logout(req: RefreshRequest, db: Session = Depends(get_db)):
    """Refresh Token을 삭제하여 로그아웃한다."""

    return logout_user(db, req)
