"""인증이 필요한 API에서 현재 로그인 사용자를 가져오는 의존성 함수."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from interview.api.core.security import decode_access_token
from interview.api.database import get_db
from interview.api.users.model import User


# Authorization: Bearer 토큰을 읽기 위한 객체
bearer_scheme = HTTPBearer()


def authenticate_access_token(token: str, db: Session) -> User:
    """Access Token을 검증하고 해당 서비스 사용자를 반환한다.

    HTTP Bearer 의존성과 WebSocket 첫 인증 메시지가 같은 JWT·사용자 조회
    규칙을 사용하도록 토큰 문자열 이후의 검증을 공통 함수로 분리한다.

    Args:
        token:
            로그인 또는 refresh API에서 발급한 Access Token 문자열.

        db:
            토큰의 사용자 이메일로 User를 조회할 SQLAlchemy 세션.

    Returns:
        토큰의 ``sub`` 이메일과 일치하는 서비스 사용자.

    Raises:
        HTTPException:
            토큰이 유효하지 않거나 사용자 정보가 없거나 DB 사용자를 찾지
            못한 경우 401 오류.
    """
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


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """HTTP Bearer Access Token으로 현재 로그인 사용자를 반환한다.

    Args:
        credentials:
            Authorization 헤더에서 읽은 Bearer 인증 정보.

        db:
            인증된 사용자를 조회할 SQLAlchemy 세션.

    Returns:
        Access Token에 연결된 현재 서비스 사용자.
    """
    return authenticate_access_token(credentials.credentials, db)
