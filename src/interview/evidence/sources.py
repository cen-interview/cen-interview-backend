"""외부 소스 접근 (Notion MCP / GitHub MCP).

인덱싱 파이프라인이 원본을 긁어오는 부분만 담당한다. 추출/청킹/저장은 각각
extract / chunking / store 가 맡는다 (한 파일에 다 넣지 않는다).

⚠️ 면접이 시작되면 이 모듈은 동작하지 않는다. 면접 전 1회만 호출.
"""

from dataclasses import dataclass


@dataclass
class RawDoc:
    """긁어온 원본 문서 1건 (추출 전)."""

    source_url: str
    source_type: str          # "notion" | "github"
    title: str
    raw_text: str
    meta: dict                # 주차/날짜/파일경로 등 소스가 아는 정보


class NotionSource:
    """Notion MCP 로 학습 기록을 가져온다."""

    def fetch_pages(self, root_link: str) -> list[RawDoc]:
        """입력 링크가 DB / 주차 페이지 / 개별 페이지 중 무엇인지 판단하고,
        하위 페이지까지 재귀 탐색해 RawDoc 리스트로 반환한다.

        TODO(담당 A):
          - 링크 유형 판별 (DB vs page)
          - 하위 페이지 재귀 순회
          - 주차/날짜/문서유형을 meta 에 채우기
        """
        raise NotImplementedError


class GitHubSource:
    """GitHub MCP 로 프로젝트 저장소(최대 3개)를 가져온다."""

    MAX_REPOS = 3

    def fetch_repos(self, repo_links: list[str]) -> list[RawDoc]:
        """저장소 접근 가능 여부, README, 디렉터리 구조, 주요 언어/프레임워크,
        핵심 구현 파일을 확인해 RawDoc 리스트로 반환한다.

        TODO(담당 A):
          - 최대 3개 제한 (초과 시 앞 3개만)
          - README / 핵심 구현 파일 선별
          - 언어/프레임워크 식별 → meta
        """
        if len(repo_links) > self.MAX_REPOS:
            repo_links = repo_links[: self.MAX_REPOS]
        raise NotImplementedError
