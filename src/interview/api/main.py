"""FastAPI 진입점.

실행:
    uv run uvicorn interview.api.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interview.api.database import Base, engine

# 모델 import: create_all이 테이블 정보를 알 수 있게 하기 위함
from interview.api.users.model import User
from interview.api.auth.model import RefreshToken

from interview.api.users.router import router as users_router
from interview.api.auth.router import router as auth_router

from dotenv import load_dotenv
from openai import OpenAI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 SQLAlchemy 모델 기준으로 없는 테이블 자동 생성
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Interview Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 회원 관련 API
app.include_router(users_router, prefix="/api")

# 인증 관련 API
app.include_router(auth_router, prefix="/api")

# 임시 ======================
load_dotenv()
client = OpenAI()


@app.post("/api/interview/realtime-transcription/token")
def create_realtime_transcription_token():
    token = client.realtime.client_secrets.create(
        expires_after={
            "anchor": "created_at",
            "seconds": 60,
        },
        session={
            "type": "transcription",
            "audio": {
                "input": {
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                        "language": "ko",
                        "delay": "high",
                    },
                    "turn_detection": None,
                },
            },
        },
    )

    return {
        "value": token.value,
        "expires_at": token.expires_at,
    }

@app.get("/api/health")
def health():
    return {"status": "ok"}
