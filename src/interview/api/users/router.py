"""회원 관련 API 라우터."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from interview.api.auth.dependency import get_current_user
from interview.api.database import get_db
from interview.api.users.model import User
from interview.api.users.schema import UserCreate, UserResponse
from interview.api.users.service import create_user

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user_create: UserCreate, db: Session = Depends(get_db)):
    """새 회원을 등록한다."""

    return create_user(db, user_create)


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """현재 로그인한 사용자의 정보를 조회한다."""

    return current_user
