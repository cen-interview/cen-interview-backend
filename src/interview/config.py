"""앱 전역 설정.

.env 파일과 환경변수에서 설정을 읽어온다. 모든 모듈은 여기서 `settings`를
import 해서 쓰고, 절대 os.environ을 직접 만지지 않는다 (설정 출처를 한 곳으로 유지).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-5.4-mini"

    # MCP / 외부 소스
    notion_api_key: str = ""
    github_token: str = ""

    # Vector DB (Postgres + pgvector)
    database_url: str = "postgresql://postgres:postgres@localhost:5432/interview"

    # 면접 진행
    max_questions: int = 10


@lru_cache
def get_settings() -> Settings:
    """설정 싱글톤. lru_cache 로 한 번만 로딩한다."""
    return Settings()


settings = get_settings()
