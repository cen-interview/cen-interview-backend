from interview.api.database import Base, engine

# 중요: 모델 파일을 import 해야 SQLAlchemy가 User 모델을 인식함
from interview.api.users.model import User

# 나중에 지울 파일임
# 테이블 생성 확인용 테스트 파일

Base.metadata.create_all(bind=engine)

print("테이블 생성 완료")