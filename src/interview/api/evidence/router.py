"""Evidence 관련 API 라우터.

사용자별 OAuth credential을 사용해 외부 MCP 연결 상태와 tool 목록을 확인한다.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from interview.api.auth.dependency import get_current_user
from interview.api.auth.model import GitHubCredential, NotionCredential
from interview.api.database import get_db
from interview.api.users.model import User
from interview.evidence.mcp_client import EvidenceMcpClient


router = APIRouter(prefix="/evidence", tags=["Evidence"])

def _format_exception(exc: BaseException) -> str:
      """ExceptionGroup 내부 원인까지 펼쳐 디버깅 메시지로 변환한다."""
      if isinstance(exc, BaseExceptionGroup):
          return " | ".join(_format_exception(child) for child in exc.exceptions)

      return f"{type(exc).__name__}: {exc}"

@router.get("/notion/tools")
def list_notion_mcp_tools(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """현재 사용자의 Notion credential로 MCP tool 목록을 조회한다.

    이 endpoint는 공식 Notion MCP 서버가 제공하는 tool 이름과 input schema를
    확인해 수집 로직이 사용할 tool 계약을 검증한다.
    """
    credential = (
         db.query(NotionCredential)
         .filter(NotionCredential.user_id == current_user.id)
       .first()
    )

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

    credential = (
         db.query(GitHubCredential)
         .filter(GitHubCredential.user_id == current_user.id)
       .first()
    )

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
