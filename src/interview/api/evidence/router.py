"""Evidence 관련 API 라우터.

사용자별 OAuth credential을 사용해 외부 MCP 연결 상태와 tool 목록을 확인하고,
등록된 Notion/GitHub 링크의 백그라운드 인덱싱을 시작한다.
"""

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from interview.api.auth.dependency import get_current_user
from interview.api.auth.model import GitHubCredential, NotionCredential
from interview.api.database import get_db
from interview.api.users.model import User
from interview.evidence.indexing import build_index
from interview.evidence.mcp_client import EvidenceMcpClient
from interview.evidence.sources import GitHubSource, NotionSource
from interview.schemas.evidence import CoverageMap, IndexBuildResult, IndexFailure


router = APIRouter(prefix="/evidence", tags=["Evidence"])


class EvidenceIndexRequest(BaseModel):
    """사용자가 등록한 외부 링크 기반 인덱싱 요청."""

    notion_links: list[str] = Field(default_factory=list)
    github_links: list[str] = Field(default_factory=list)


class EvidenceIndexStatus(BaseModel):
    """사용자별 Evidence 인덱싱 진행 상태."""

    status: Literal["idle", "running", "success", "partial_failed", "failed"]
    user_id: int
    started_at: str | None = None
    updated_at: str | None = None
    result: IndexBuildResult | None = None


_index_status_by_user: dict[int, EvidenceIndexStatus] = {}


def _format_exception(exc: BaseException) -> str:
    """ExceptionGroup 내부 원인까지 펼쳐 디버깅 메시지로 변환한다."""
    if isinstance(exc, BaseExceptionGroup):
        return " | ".join(_format_exception(child) for child in exc.exceptions)

    return f"{type(exc).__name__}: {exc}"


def _now_iso() -> str:
    """API 상태 응답에 사용할 UTC ISO timestamp를 반환한다."""
    return datetime.now(timezone.utc).isoformat()


def _get_notion_credential(
    db: Session,
    user_id: int,
) -> NotionCredential | None:
    """사용자별 Notion credential을 조회한다."""
    return (
        db.query(NotionCredential)
        .filter(NotionCredential.user_id == user_id)
        .first()
    )


def _get_github_credential(
    db: Session,
    user_id: int,
) -> GitHubCredential | None:
    """사용자별 GitHub credential을 조회한다."""
    return (
        db.query(GitHubCredential)
        .filter(GitHubCredential.user_id == user_id)
        .first()
    )


def _run_index_background(
    *,
    user_id: int,
    notion_links: list[str],
    github_links: list[str],
    notion_access_token: str | None,
    github_access_token: str | None,
    github_login: str | None,
) -> None:
    """사용자별 credential이 주입된 source로 Evidence 인덱싱을 실행한다."""
    try:
        mcp_client = EvidenceMcpClient(
            notion_access_token=notion_access_token,
            github_access_token=github_access_token,
        )
        result = build_index(
            notion_links=notion_links,
            github_links=github_links,
            user_id=user_id,
            notion_source=NotionSource(mcp_client=mcp_client),
            github_source=GitHubSource(mcp_client=mcp_client),
            github_login=github_login,
        )
    except Exception as exc:
        result = IndexBuildResult(
            status="failed",
            coverage_map=CoverageMap(),
            raw_doc_count=0,
            chunk_count=0,
            failures=[
                IndexFailure(
                    source_type="evidence",
                    source_url=None,
                    stage="index",
                    message=_format_exception(exc),
                )
            ],
        )

    previous_status = _index_status_by_user.get(user_id)
    _index_status_by_user[user_id] = EvidenceIndexStatus(
        status=result.status,
        user_id=user_id,
        started_at=previous_status.started_at if previous_status else None,
        updated_at=_now_iso(),
        result=result,
    )


@router.post("/index")
def start_evidence_index(
    request: EvidenceIndexRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EvidenceIndexStatus:
    """등록된 Notion/GitHub 링크를 사용자별 Evidence store로 백그라운드 인덱싱한다."""
    if not request.notion_links and not request.github_links:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="인덱싱할 Notion 또는 GitHub 링크가 필요합니다.",
        )

    current_status = _index_status_by_user.get(current_user.id)
    if current_status is not None and current_status.status == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="이미 인덱싱이 진행 중입니다.",
        )

    notion_credential = _get_notion_credential(db, current_user.id)
    github_credential = _get_github_credential(db, current_user.id)

    if request.notion_links and notion_credential is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notion 연결 정보가 없습니다.",
        )

    if request.github_links and github_credential is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="GitHub 연결 정보가 없습니다.",
        )

    now = _now_iso()
    status_payload = EvidenceIndexStatus(
        status="running",
        user_id=current_user.id,
        started_at=now,
        updated_at=now,
        result=None,
    )
    _index_status_by_user[current_user.id] = status_payload

    background_tasks.add_task(
        _run_index_background,
        user_id=current_user.id,
        notion_links=request.notion_links,
        github_links=request.github_links,
        notion_access_token=(
            notion_credential.mcp_access_token if notion_credential else None
        ),
        github_access_token=(
            github_credential.access_token if github_credential else None
        ),
        github_login=github_credential.github_login if github_credential else None,
    )

    return status_payload


@router.get("/status")
def get_evidence_index_status(
    current_user: User = Depends(get_current_user),
) -> EvidenceIndexStatus:
    """현재 사용자의 최근 Evidence 인덱싱 상태를 반환한다."""
    return _index_status_by_user.get(
        current_user.id,
        EvidenceIndexStatus(status="idle", user_id=current_user.id),
    )


@router.get("/notion/tools")
def list_notion_mcp_tools(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """현재 사용자의 Notion credential로 MCP tool 목록을 조회한다.

    이 endpoint는 공식 Notion MCP 서버가 제공하는 tool 이름과 input schema를
    확인해 수집 로직이 사용할 tool 계약을 검증한다.
    """
    credential = _get_notion_credential(db, current_user.id)

    if credential is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notion 연결 정보가 없습니다.",
        )

    try:
        tools = EvidenceMcpClient(
            notion_access_token=credential.mcp_access_token,
        ).list_notion_tools()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Notion MCP tool 조회 실패: {_format_exception(exc)}",
        ) from exc

    return {"tools": tools}


@router.get("/github/tools")
def list_github_mcp_tools(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """현재 사용자의 GitHub credential로 MCP tool 목록을 조회한다."""

    credential = _get_github_credential(db, current_user.id)

    if credential is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="GitHub 연결 정보가 없습니다.",
        )

    try:
        tools = EvidenceMcpClient(
            github_access_token=credential.access_token,
        ).list_github_tools()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"GitHub MCP tool 조회 실패: {_format_exception(exc)}",
        ) from exc

    return {"tools": tools}
