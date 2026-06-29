# DB 컬럼을 정의할 때 사용하는 기본 타입들
from sqlalchemy import Column, Integer, String, DateTime

# DB 서버 기준 현재 시간을 넣기 위해 사용
from sqlalchemy.sql import func

# database.py에서 만든 Base 가져오기
# 모든 SQLAlchemy 모델은 이 Base를 상속받아야 함
from interview.api.database import Base


# User 테이블 모델 정의
# Spring JPA의 @Entity 클래스와 비슷한 역할
class User(Base):

    # 실제 DB에 생성될 테이블 이름
    __tablename__ = "users"

    # 회원 고유 ID
    # primary_key=True: 기본키
    # index=True: 검색 성능을 위한 인덱스 생성
    id = Column(Integer, primary_key=True, index=True)

    # 이메일 컬럼
    # String(255): 최대 255자 문자열
    # unique=True: 중복 이메일 허용 안 함
    # nullable=False: 필수값
    # index=True: 이메일 검색을 빠르게 하기 위한 인덱스
    email = Column(String(255), unique=True, nullable=False, index=True)

    # 비밀번호 컬럼
    # 실제 비밀번호를 그대로 저장하면 안 되고,
    # 암호화된 비밀번호를 저장해야 함
    password = Column(String(255), nullable=False)

    # 사용자 이름 컬럼
    # 필수값으로 설정
    name = Column(String(50), nullable=False)

    # 회원가입 시간 컬럼
    # server_default=func.now(): DB 서버 기준 현재 시간이 자동 저장됨
    created_at = Column(DateTime(timezone=True), server_default=func.now())