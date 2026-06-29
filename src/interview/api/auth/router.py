"""인증 관련 API 라우터."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from interview.api.auth.model import RefreshToken
from interview.api.auth.schema import LoginRequest, TokenResponse, RefreshRequest
from interview.api.core.security import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    verify_password,
)
from interview.api.database import get_db
from interview.api.users.model import User


router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """로그인 후 Access Token과 Refresh Token을 발급한다."""

    # 이메일로 사용자 조회
    user = db.query(User).filter(User.email == req.email).first()

    # 사용자가 없거나 비밀번호가 틀리면 로그인 실패
    if not user or not verify_password(req.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다.",
        )

    # Access Token 생성
    access_token = create_access_token(
        data={
            "sub": user.email,
            "user_id": user.id,
        }
    )

    # Refresh Token 생성
    refresh_token = create_refresh_token(
        data={
            "sub": user.email,
            "user_id": user.id,
        }
    )

    # Refresh Token 만료 시간 설정
    refresh_expires_at = datetime.now(timezone.utc) + timedelta(days=7)

    # Refresh Token DB 저장
    db_refresh_token = RefreshToken(
        token=refresh_token,
        user_id=user.id,
        expires_at=refresh_expires_at,
    )

    db.add(db_refresh_token)
    db.commit()

    # Access Token + Refresh Token 반환
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )

@router.post("/refresh")
def refresh_access_token(req: RefreshRequest, db: Session = Depends(get_db)):
    """Refresh Token을 검증하고 새로운 Access Token을 발급한다."""

    # Refresh Token JWT 검증
    payload = decode_access_token(req.refresh_token)

    # 토큰이 잘못되었거나 만료된 경우
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="유효하지 않은 Refresh Token입니다.",
        )

    # payload에서 사용자 정보 추출
    email = payload.get("sub")
    user_id = payload.get("user_id")

    # 토큰 안에 필요한 정보가 없는 경우
    if email is None or user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Refresh Token 정보가 올바르지 않습니다.",
        )

    # DB에 저장된 Refresh Token인지 확인
    db_refresh_token = (
        db.query(RefreshToken)
        .filter(RefreshToken.token == req.refresh_token)
        .first()
    )

    # DB에 없으면 로그아웃되었거나 위조된 토큰으로 판단
    if db_refresh_token is None:
        raise HTTPException(
            status_code=401,
            detail="저장되지 않은 Refresh Token입니다.",
        )

    # Refresh Token 만료 시간 확인
    if db_refresh_token.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=401,
            detail="만료된 Refresh Token입니다.",
        )

    # 사용자 조회
    user = db.query(User).filter(User.id == user_id, User.email == email).first()

    # 사용자가 존재하지 않는 경우
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="사용자를 찾을 수 없습니다.",
        )

    # 새로운 Access Token 생성
    new_access_token = create_access_token(
        data={
            "sub": user.email,
            "user_id": user.id,
        }
    )

    # 새 Access Token 반환
    return {
        "access_token": new_access_token,
        "token_type": "bearer",
    }

@router.post("/logout")
def logout(req: RefreshRequest, db: Session = Depends(get_db)):
    """Refresh Token을 삭제하여 로그아웃 처리한다."""

    # DB에 저장된 Refresh Token 조회
    db_refresh_token = (
        db.query(RefreshToken)
        .filter(RefreshToken.token == req.refresh_token)
        .first()
    )

    # 이미 로그아웃되었거나 존재하지 않는 토큰인 경우
    if db_refresh_token is None:
        raise HTTPException(
            status_code=404,
            detail="이미 로그아웃되었거나 존재하지 않는 Refresh Token입니다.",
        )

    # Refresh Token 삭제
    db.delete(db_refresh_token)
    db.commit()

    return {
        "message": "로그아웃이 완료되었습니다."
    }