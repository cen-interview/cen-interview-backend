"""외부 소스 접근 (Notion MCP / GitHub MCP).

인덱싱 파이프라인이 원본을 긁어오는 부분만 담당한다. 추출/청킹/저장은 각각
extract / chunking / store 가 맡는다 (한 파일에 다 넣지 않는다).

⚠️ 면접이 시작되면 이 모듈은 동작하지 않는다. 면접 전 1회만 호출.
"""

from dataclasses import dataclass

from interview.evidence.mcp_client import EvidenceMcpClient


@dataclass
class RawDoc:
    """긁어온 원본 문서 1건 (추출 전)."""

    source_url: str
    source_type: str          # "notion" | "github"
    title: str
    raw_text: str
    meta: dict                # 주차/날짜/파일경로 등 소스가 아는 정보


def _notion_response_to_raw_docs(response: dict, root_link: str) -> list[RawDoc]:
    """Notion MCP 원본 응답을 Evidence 파이프라인의 RawDoc 목록으로 변환한다.

    Args:
        response: EvidenceMcpClient가 Notion MCP tool call로 받은 원본 응답.
        root_link: 사용자가 등록한 Notion 루트 링크.

    Returns:
        extract 단계로 넘길 Notion RawDoc 목록.
    """
    raise NotImplementedError


def _github_response_to_raw_docs(response: dict, repo_link: str) -> list[RawDoc]:
    """GitHub MCP 원본 응답을 Evidence 파이프라인의 RawDoc 목록으로 변환한다.

    Args:
        response: EvidenceMcpClient가 GitHub MCP tool call로 받은 원본 응답.
        repo_link: 사용자가 등록한 GitHub 저장소 링크.

    Returns:
        README, 디렉터리 구조, 핵심 소스 파일 등을 담은 GitHub RawDoc 목록.
    """
    raise NotImplementedError


class NotionSource:
    """Notion MCP 로 사용자가 등록한 학습 기록 링크 목록을 가져온다."""

    def __init__(self, mcp_client: EvidenceMcpClient | None = None) -> None:
        """MCP client를 주입받아 테스트와 실제 tool call 경계를 분리한다."""
        self.mcp_client = mcp_client or EvidenceMcpClient()

    def fetch_pages(self, root_links: list[str]) -> list[RawDoc]:
        """등록된 Notion 링크를 순회하며 RawDoc 리스트로 반환한다.

        각 입력 링크가 DB / 주차 페이지 / 개별 페이지 중 무엇인지 판단하고,
        하위 페이지까지 재귀 탐색한다.

        TODO(담당 A):
          - 등록된 링크를 순서대로 순회
          - 링크 유형 판별 (DB vs page)
          - 하위 페이지 재귀 순회
          - 주차/날짜/문서유형을 meta 에 채우기
        """
        raw_docs: list[RawDoc] = []
        for root_link in root_links:
            response = self.mcp_client.call_notion_tool(root_link)
            raw_docs.extend(_notion_response_to_raw_docs(response, root_link))
        return raw_docs


class GitHubSource:
    """GitHub MCP 로 사용자가 등록한 프로젝트 저장소 목록을 가져온다."""

    def __init__(self, mcp_client: EvidenceMcpClient | None = None) -> None:
        """MCP client를 주입받아 테스트와 실제 tool call 경계를 분리한다."""
        self.mcp_client = mcp_client or EvidenceMcpClient()

    def fetch_repos(self, repo_links: list[str]) -> list[RawDoc]:
        """저장소 접근 가능 여부, README, 디렉터리 구조, 주요 언어/프레임워크,
        핵심 구현 파일을 확인해 RawDoc 리스트로 반환한다.

        TODO(담당 A):
          - 등록된 링크를 순서대로 순회
          - README / 핵심 구현 파일 선별
          - 언어/프레임워크 식별 → meta
        """
        raw_docs: list[RawDoc] = []
        for repo_link in repo_links:
            response = self.mcp_client.call_github_tool(repo_link)
            raw_docs.extend(_github_response_to_raw_docs(response, repo_link))
        return raw_docs
