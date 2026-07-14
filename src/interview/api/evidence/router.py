"""Evidence 관련 API 라우터.

사용자별 OAuth credential을 사용해 외부 MCP 연결 상태와 tool 목록을 확인하고,
등록된 Notion/GitHub 링크의 백그라운드 인덱싱을 시작한다.
"""

from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from interview.api.auth.dependency import get_current_user
from interview.api.auth.model import GitHubCredential, NotionCredential
from interview.api.database import get_db
from interview.api.evidence.model import EvidenceSourceLink
from interview.api.evidence.schema import (
    EvidenceIndexRequest,
    EvidenceIndexStatus,
    EvidenceSourceCreateRequest,
    EvidenceSourceListResponse,
    EvidenceSourceResponse,
    EvidenceSummaryResponse,
)
from interview.api.users.model import User
from interview.evidence.indexing import build_index
from interview.evidence.mcp_client import EvidenceMcpClient
from interview.evidence.sources import GitHubSource, NotionSource
from interview.evidence.store import get_store
from interview.schemas.evidence import CoverageMap, IndexBuildResult, IndexFailure


router = APIRouter(prefix="/evidence", tags=["Evidence"])


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


def _normalize_source_url(source_type: str, url: str) -> str:
    """같은 자료가 중복 등록되지 않도록 URL을 정규화하고 출처를 검증한다."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    allowed_hosts = {
        "github": {"github.com", "www.github.com"},
        "notion": {"notion.so", "www.notion.so", "app.notion.com"},
    }

    if parsed.scheme not in {"http", "https"} or host not in allowed_hosts[source_type]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{source_type} 형식의 공개 링크가 필요합니다.",
        )

    path = parsed.path.rstrip("/")
    if source_type == "github" and path.endswith(".git"):
        path = path[:-4]
    if not path or path == "/":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="저장소 또는 페이지 경로가 포함된 링크가 필요합니다.",
        )

    return urlunparse(("https", host, path, "", "", ""))


def _source_response(source: EvidenceSourceLink) -> EvidenceSourceResponse:
    """ORM 링크 모델을 프론트엔드 응답 모델로 변환한다."""

    return EvidenceSourceResponse(
        id=source.id,
        source_type=source.source_type,
        url=source.url,
        normalized_url=source.normalized_url,
        created_at=source.created_at,
        updated_at=source.updated_at,
    )


def _links_from_request(
    request: EvidenceIndexRequest,
    current_user: User,
    db: Session,
) -> tuple[list[str], list[str]]:
    """직접 전달 링크와 저장된 링크 ID를 인덱싱용 출처 목록으로 합친다."""

    notion_links = list(request.notion_links)
    github_links = list(request.github_links)

    if request.source_ids:
        sources = (
            db.query(EvidenceSourceLink)
            .filter(
                EvidenceSourceLink.user_id == current_user.id,
                EvidenceSourceLink.id.in_(request.source_ids),
            )
            .all()
        )
        found_ids = {source.id for source in sources}
        missing_ids = sorted(set(request.source_ids) - found_ids)
        if missing_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"등록된 Evidence 링크를 찾을 수 없습니다: {missing_ids}",
            )
        notion_links.extend(
            source.url for source in sources if source.source_type == "notion"
        )
        github_links.extend(
            source.url for source in sources if source.source_type == "github"
        )

    if not notion_links and not github_links and not request.source_ids:
        sources = (
            db.query(EvidenceSourceLink)
            .filter(EvidenceSourceLink.user_id == current_user.id)
            .all()
        )
        notion_links = [source.url for source in sources if source.source_type == "notion"]
        github_links = [source.url for source in sources if source.source_type == "github"]

    return list(dict.fromkeys(notion_links)), list(dict.fromkeys(github_links))


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


@router.get("/sources", response_model=EvidenceSourceListResponse)
def list_evidence_sources(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EvidenceSourceListResponse:
    """마이페이지에 표시할 현재 사용자의 등록 자료 링크를 반환한다."""

    sources = (
        db.query(EvidenceSourceLink)
        .filter(EvidenceSourceLink.user_id == current_user.id)
        .order_by(EvidenceSourceLink.created_at.desc())
        .all()
    )
    return EvidenceSourceListResponse(sources=[_source_response(source) for source in sources])


@router.post("/sources", response_model=EvidenceSourceResponse, status_code=status.HTTP_201_CREATED)
def create_evidence_source(
    request: EvidenceSourceCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EvidenceSourceResponse:
    """마이페이지에서 입력한 GitHub 또는 Notion 링크를 저장한다."""

    source_type = request.source_type
    url = str(request.url)
    normalized_url = _normalize_source_url(source_type, url)
    source = (
        db.query(EvidenceSourceLink)
        .filter(
            EvidenceSourceLink.user_id == current_user.id,
            EvidenceSourceLink.normalized_url == normalized_url,
        )
        .first()
    )
    if source is None:
        source = EvidenceSourceLink(
            user_id=current_user.id,
            source_type=source_type,
            url=url,
            normalized_url=normalized_url,
        )
        db.add(source)
        db.commit()
        db.refresh(source)
        return _source_response(source)

    return _source_response(source)


@router.delete("/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_evidence_source(
    source_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """사용자가 등록한 링크를 제거한다.

    이미 저장된 벡터 청크는 다음 인덱싱에서 해당 출처 전체를 재구축하며 정리된다.
    """

    source = (
        db.query(EvidenceSourceLink)
        .filter(
            EvidenceSourceLink.id == source_id,
            EvidenceSourceLink.user_id == current_user.id,
        )
        .first()
    )
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="등록된 Evidence 링크를 찾을 수 없습니다.",
        )
    db.delete(source)
    db.commit()


@router.post("/index")
def start_evidence_index(
    request: EvidenceIndexRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EvidenceIndexStatus:
    """등록된 Notion/GitHub 링크를 사용자별 Evidence store로 백그라운드 인덱싱한다."""
    full_reindex = not (
        request.notion_links or request.github_links or request.source_ids
    )
    notion_links, github_links = _links_from_request(request, current_user, db)
    if not notion_links and not github_links:
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

    if notion_links and notion_credential is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notion 연결 정보가 없습니다.",
        )

    if github_links and github_credential is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="GitHub 연결 정보가 없습니다.",
        )

    if full_reindex:
        # 저장된 링크 전체를 다시 읽는 요청은 삭제된 자료의 이전 청크도 제거한다.
        get_store().clear_user_sources({"notion", "github"}, user_id=current_user.id)

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
        notion_links=notion_links,
        github_links=github_links,
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


@router.get("/summary", response_model=EvidenceSummaryResponse)
def get_evidence_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EvidenceSummaryResponse:
    """마이페이지 분석 카드에 필요한 링크 수와 현재 인덱싱 요약을 반환한다."""

    source_counts = {"github": 0, "notion": 0}
    sources = (
        db.query(EvidenceSourceLink.source_type)
        .filter(EvidenceSourceLink.user_id == current_user.id)
        .all()
    )
    for (source_type,) in sources:
        if source_type in source_counts:
            source_counts[source_type] += 1

    index_status = _index_status_by_user.get(
        current_user.id,
        EvidenceIndexStatus(status="idle", user_id=current_user.id),
    )
    result = index_status.result
    coverage_map = (
        result.coverage_map
        if result is not None
        else get_store().build_coverage_map(user_id=current_user.id)
    )
    return EvidenceSummaryResponse(
        index_status=index_status.status,
        last_indexed_at=index_status.updated_at,
        source_counts=source_counts,
        raw_doc_count=result.raw_doc_count if result else 0,
        chunk_count=result.chunk_count if result else 0,
        coverage_map=coverage_map,
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
