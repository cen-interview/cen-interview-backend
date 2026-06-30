# Pydantic의 BaseModel을 상속하면 요청/응답 데이터 검증용 클래스를 만들 수 있음
# EmailStr은 이메일 형식인지 검증해주는 타입
# ConfigDict는 Pydantic 설정을 지정할 때 사용
from pydantic import BaseModel, EmailStr, ConfigDict


# 회원가입 요청을 받을 때 사용하는 Schema
# 스프링으로 치면 회원가입 Request DTO 역할
class UserCreate(BaseModel):
    # 사용자가 입력한 이메일
    # EmailStr을 사용하면 이메일 형식이 아니면 자동으로 검증 에러가 발생함
    email: EmailStr

    # 사용자가 입력한 비밀번호
    # 실제 DB에는 이 값을 그대로 저장하지 않고 암호화해서 저장해야 함
    password: str

    # 사용자 이름
    name: str


# 회원 정보를 응답으로 내려줄 때 사용하는 Schema
# 스프링으로 치면 Response DTO 역할
class UserResponse(BaseModel):
    # DB에 저장된 사용자 고유 ID
    id: int

    # 사용자 이메일
    email: EmailStr

    # 사용자 이름
    name: str

    # SQLAlchemy ORM 객체를 Pydantic 응답 객체로 변환할 수 있게 해주는 설정
    # 예: User 모델 객체를 UserResponse로 자동 변환 가능
    model_config = ConfigDict(from_attributes=True)