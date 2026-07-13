"""앱 전역 설정.

.env 파일과 환경변수에서 설정을 읽어온다. 모든 모듈은 여기서 `settings`를
import 해서 쓰고, 절대 os.environ을 직접 만지지 않는다 (설정 출처를 한 곳으로 유지).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수와 .env 파일에서 읽는 애플리케이션 설정 계약."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini" # "gpt-5.4-mini"

    # MCP / 외부 소스
    # notion
    notion_mcp_url: str = "https://mcp.notion.com/mcp"
    notion_mcp_issuer: str = "https://mcp.notion.com"
    notion_mcp_resource: str = "https://mcp.notion.com/mcp"
    notion_redirect_uri: str = "http://localhost:8000/api/auth/notion/callback"
    notion_mcp_access_token: str = ""

    # github
    github_token: str = ""
    github_oauth_client_id: str = ""
    github_oauth_client_secret: str = ""
    github_oauth_redirect_uri: str = "http://localhost:8000/api/auth/github/callback"
    github_oauth_scope: str = "read:user repo"
    github_mcp_url: str = ""
    github_mcp_access_token: str = ""
    github_mcp_repository_tool: str = "search_repositories"
    github_mcp_contents_tool: str = "get_file_contents"
    github_mcp_commits_tool: str = "list_commits"
    github_mcp_commit_tool: str = "get_commit"

    # Vector DB (Postgres + pgvector)
    database_url: str = "postgresql://postgres:postgres@localhost:5432/interview"

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536   # 차원 지정 1536

    # 면접 진행
    max_questions: int = 10
    
    # Evidence
    use_stub_evidence: bool = False
    evidence_llm_extract_enabled: bool = False
    evidence_store_backend: str = "memory"
    evidence_chroma_path: str = ".chroma/evidence"


@lru_cache
def get_settings() -> Settings:
    """설정 싱글톤. lru_cache 로 한 번만 로딩한다."""
    return Settings()

settings = get_settings()
