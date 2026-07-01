"""인증 관련 비즈니스 로직."""

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from interview.api.auth.model import RefreshToken
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
