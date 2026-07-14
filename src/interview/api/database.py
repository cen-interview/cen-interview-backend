import os

# SQLAlchemy에서 DB 연결 엔진을 만들 때 사용하는 함수
from sqlalchemy import create_engine

# DB 세션을 만들기 위한 sessionmaker,
# ORM 모델들의 부모 클래스 역할을 하는 declarative_base를 가져옴
from sqlalchemy.orm import sessionmaker, declarative_base

from dotenv import load_dotenv # DB_URL 호출 필요

load_dotenv()

# PostgreSQL DB 접속 주소
# 형식: postgresql+psycopg://사용자명:비밀번호@호스트:포트/DB이름
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://interview:1234@localhost:5432/interviewdb",  # 로컬 개발 fallback
)

# 실제 DB와 연결하는 엔진 생성
# SQLAlchemy가 이 엔진을 통해 PostgreSQL과 통신함
engine = create_engine(DATABASE_URL)


# DB 작업을 할 때 사용할 세션 생성기
# 세션은 DB와 대화하는 작업 단위라고 보면 됨
SessionLocal = sessionmaker(
    autocommit=False,  # 자동 커밋 비활성화. 직접 commit 해야 DB에 반영됨
    autoflush=False,   # 자동 flush 비활성화. 필요할 때 명시적으로 반영
    bind=engine        # 위에서 만든 DB 엔진과 연결
)


# ORM 모델 클래스들의 부모 클래스
# 앞으로 만드는 User 같은 모델은 이 Base를 상속받아야 함
Base = declarative_base()


def create_missing_tables() -> None:
    """SQLAlchemy 모델에 정의됐지만 DB에 없는 테이블을 생성한다.

    FastAPI 애플리케이션 시작 시 호출하는 초기화 함수다. 각 모델 모듈을
    먼저 import해 모든 테이블을 Base.metadata에 등록한 다음 create_all을
    실행한다. 이미 존재하는 테이블과 데이터는 변경하지 않는다.

    Returns:
        None.
    """
    from interview.api.auth import model as auth_model
    from interview.api.interviews import model as interviews_model
    from interview.api.users import model as users_model

    _ = (auth_model, interviews_model, users_model)
    Base.metadata.create_all(bind=engine)

# FastAPI에서 DB 세션을 의존성 주입으로 사용하기 위한 함수
# API 요청이 들어올 때 DB 세션을 하나 만들고,
# 요청 처리가 끝나면 세션을 닫아줌
def get_db():
    # DB 세션 생성
    db = SessionLocal()

    try:
        # router 함수에 db 세션을 전달
        yield db

    finally:
        # 요청 처리가 끝나면 DB 세션 닫기
        db.close()
