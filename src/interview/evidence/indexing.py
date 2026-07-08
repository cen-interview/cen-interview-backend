"""인덱싱 파이프라인 (면접 전 1회 실행).

분기·루프가 없는 고정 파이프라인이라 에이전트가 아니다. sources → extract →
chunking → store 순서를 배선만 한다.

    fetch(Notion/GitHub) → extract → chunk → store.add → coverage map

이 함수가 끝나면 evidence_store 가 준비되고, 이후 면접 중에는 retrieval 만 쓴다.
"""

from interview.evidence.chunking import chunk
from interview.evidence.extract import extract_evidence
from interview.evidence.sources import GitHubSource, NotionSource
from interview.evidence.store import get_store
from interview.schemas.evidence import CoverageMap


def build_index(
    notion_link: str,
    github_links: list[str],
    user_id: int | str | None = None,
) -> CoverageMap:
    """면접용 지식 베이스를 구축하고 커버리지 맵을 반환한다.

    Args:
        notion_link: 사용자가 등록한 Notion 루트 링크.
        github_links: GitHub 프로젝트 링크 (최대 3개).
        user_id: Evidence store에서 사용자별 청크를 분리하기 위한 사용자 ID.
            기존 호출처럼 None이면 store의 기본 namespace에 저장한다.

    Returns:
        주제별 커버리지 맵 (Strategy 가 약한 주제 파악에 사용).
    """
    raw_docs = []
    raw_docs += NotionSource().fetch_pages(notion_link)
    raw_docs += GitHubSource().fetch_repos(github_links)

    all_chunks = []
    for doc in raw_docs:
        all_chunks += extract_evidence(doc)
    all_chunks = chunk(all_chunks)

    store = get_store()
    store.add_chunks(all_chunks, user_id=user_id)
    return store.build_coverage_map(user_id=user_id)
