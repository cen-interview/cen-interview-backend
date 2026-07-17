"""FastAPI 애플리케이션을 생성하고 기능별 라우터를 등록하는 진입점.

실행:
    uv run uvicorn interview.api.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interview.api.auth.router import router as auth_router
from interview.api.database import create_missing_tables
from interview.api.evidence.router import router as evidence_router
from interview.api.interviews.router import router as interviews_router
from interview.api.sessions.router import (
    router as sessions_router,
)
from interview.api.users.router import router as users_router
from interview.api.voice.router import router as voice_router
from interview.config import settings
from interview.evidence.question_pattern_dataset import find_question_pattern_csvs
from interview.evidence.question_pattern_ingestion import ingest_question_patterns

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작과 종료 수명주기를 관리한다.

    Args:
        app:
            수명주기를 적용할 FastAPI 애플리케이션.
    """
    if settings.database_auto_create_tables:
        create_missing_tables()

    question_pattern_csvs = find_question_pattern_csvs(settings.question_pattern_data_path)
    if question_pattern_csvs:
        csv_path = question_pattern_csvs[-1]
        ingested_rows = ingest_question_patterns(csv_path)
        logger.info(
            "질문 패턴 CSV를 DB에 적재했습니다: file=%s, rows=%d",
            csv_path,
            ingested_rows,
        )
    else:
        logger.info(
            "질문 패턴 CSV가 없어 적재를 건너뜁니다: path=%s",
            settings.question_pattern_data_path,
        )
    yield


app = FastAPI(
    title="Interview Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://cen-interview-front.vercel.app",
        "https://cen-interview.site",
        "https://www.cen-interview.site",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(evidence_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(voice_router, prefix="/api")
app.include_router(
    interviews_router,
    prefix="/api",
)


@app.get("/api/health")
def health():
    """서버 프로세스의 기본 상태를 반환한다.

    Returns:
        서버가 요청을 처리할 수 있음을 나타내는 상태 dict.
    """
    return {"status": "ok"}
