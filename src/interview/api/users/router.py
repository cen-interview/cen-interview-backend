# FastAPI에서 API 라우터를 만들기 위해 APIRouter 사용
# Depends는 DB 세션 같은 의존성을 주입받을 때 사용
# HTTPException은 에러 응답을 직접 발생시킬 때 사용
# status는 HTTP 상태 코드를 깔끔하게 쓰기 위해 사용
from fastapi import APIRouter, Depends, HTTPException, status

# SQLAlchemy DB 세션 타입
from sqlalchemy.orm import Session

# DB 세션을 가져오는 함수
from interview.api.database import get_db

# User 테이블 모델
from interview.api.users.model import User

# 회원가입 요청/응답 Schema
from interview.api.users.schema import UserCreate, UserResponse

# 비밀번호 암호화 함수
from interview.api.core.security import hash_password

# 토큰에서 유저 정보 가져오기
from interview.api.auth.dependency import get_current_user

# users 관련 API를 묶는 라우터 객체
# prefix="/users" 이므로 이 파일의 API는 전부 /users로 시작함
# tags=["users"]는 Swagger 문서에서 그룹 이름으로 표시됨
router = APIRouter(prefix="/users", tags=["users"])

# 회원가입 API
# 최종 URL: POST /users/signup
# response_model=UserResponse는 응답 데이터를 UserResponse 형태로 제한함
@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user_create: UserCreate, db: Session = Depends(get_db)):
    # 이미 같은 이메일로 가입한 사용자가 있는지 조회
    existing_user = db.query(User).filter(User.email == user_create.email).first()

    # 이미 가입된 이메일이면 400 에러 발생
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 사용 중인 이메일입니다."
        )

    # 사용자가 입력한 비밀번호를 암호화
    hashed_password = hash_password(user_create.password)

    # User 모델 객체 생성
    # password 컬럼에는 평문 비밀번호가 아니라 암호화된 비밀번호를 저장해야 함
    new_user = User(
        email=user_create.email,
        password=hashed_password,
        name=user_create.name
    )

    # DB에 새 사용자 추가
    db.add(new_user)

    # DB에 실제 저장
    db.commit()

    # 저장 후 생성된 id 값을 가져오기 위해 새로고침
    db.refresh(new_user)

    # UserResponse 형태로 응답됨
    return new_user

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """현재 로그인한 사용자의 정보를 조회한다."""

    return current_user