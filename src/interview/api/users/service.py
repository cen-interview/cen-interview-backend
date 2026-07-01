"""회원 관련 비즈니스 로직."""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from interview.api.core.security import hash_password
from interview.api.users.model import User
from interview.api.users.schema import UserCreate


def create_user(db: Session, user_create: UserCreate) -> User:
    """새 회원을 생성한다."""

    existing_user = db.query(User).filter(User.email == user_create.email).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 사용 중인 이메일입니다.",
        )

    new_user = User(
        email=user_create.email,
        password=hash_password(user_create.password),
        name=user_create.name,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user
