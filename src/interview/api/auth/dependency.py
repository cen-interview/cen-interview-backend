"""인증이 필요한 API에서 현재 로그인 사용자를 가져오는 의존성 함수."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from interview.api.core.security import decode_access_token
from interview.api.database import get_db
from interview.api.users.model import User


# Authorization: Bearer 토큰을 읽기 위한 객체
bearer_scheme = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Access Token을 검증하고 현재 로그인한 User를 반환한다."""

    # Authorization 헤더에서 토큰 문자열만 꺼낸다
    token = credentials.credentials

    # 토큰 검증
    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다.",
        )

    # 토큰에서 사용자 이메일 추출
    email = payload.get("sub")

    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰에 사용자 정보가 없습니다.",
        )

    # DB에서 사용자 조회
    user = db.query(User).filter(User.email == email).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자를 찾을 수 없습니다.",
        )

    return user