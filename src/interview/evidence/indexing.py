"""인덱싱 파이프라인 (면접 전 1회 실행).

분기·루프가 없는 고정 파이프라인이라 에이전트가 아니다. sources → extract →
chunking → store 순서를 배선만 한다.

    fetch(Notion/GitHub) → extract → chunk → store.add → coverage map

이 함수가 끝나면 evidence_store 가 준비되고, 이후 면접 중에는 retrieval 만 쓴다.
일부 source/chunk/store 구현은 외부 저장소 연동 전까지 단순 구현으로 유지한다.
"""

from interview.config import settings
from interview.evidence.chunking import chunk
from interview.evidence.extract import extract_evidence, refine_evidence_chunks
from interview.evidence.sources import GitHubSource, NotionSource, RawDoc
from interview.evidence.store import get_store
from interview.schemas.evidence import IndexBuildResult, IndexFailure


def build_index(
    notion_links: list[str],
    github_links: list[str],
    user_id: int | str | None = None,
    notion_source: NotionSource | None = None,
    github_source: GitHubSource | None = None,
    github_login: str | None = None,
) -> IndexBuildResult:
    """면접용 지식 베이스를 구축하고 커버리지 맵을 반환한다.

    Args:
        notion_links: 사용자가 등록한 Notion 학습 기록 링크 목록.
        github_links: 사용자가 등록한 GitHub 프로젝트 링크 목록.
        user_id: Evidence store에서 사용자별 청크를 분리하기 위한 사용자 ID.
            None이면 store의 기본 namespace를 사용한다.
        notion_source: 사용자별 Notion credential이 주입된 source. 없으면 기본 source를 사용한다.
        github_source: 사용자별 GitHub credential이 주입된 source. 없으면 기본 source를 사용한다.
        github_login: GitHub commit 조회에서 author 필터로 사용할 사용자 login.

    Returns:
        인덱싱 상태, 실패 목록, 저장 청크 수, 주제별 커버리지 맵.
    """
    store = get_store()
    requested_sources = {
        source_type
        for source_type, links in (("notion", notion_links), ("github", github_links))
        if links
    }
    if hasattr(store, "clear_user_sources"):
        store.clear_user_sources(requested_sources, user_id=user_id)
    elif requested_sources:
        store.clear_user(user_id=user_id)

    failures: list[IndexFailure] = []
    raw_docs: list[RawDoc] = []

    notion_source = notion_source or NotionSource()
    for link in notion_links:
        try:
            raw_docs.extend(notion_source.fetch_pages([link]))
        except Exception as exc:
            failures.append(
                _failure(
                    source_type="notion",
                    source_url=link,
                    stage="fetch",
                    exc=exc,
                )
            )

    github_source = github_source or GitHubSource()
    for link in github_links:
        try:
            raw_docs.extend(github_source.fetch_repos([link], github_login=github_login))
        except Exception as exc:
            failures.append(
                _failure(
                    source_type="github",
                    source_url=link,
                    stage="fetch",
                    exc=exc,
                )
            )

    all_chunks = []
    for doc in raw_docs:
        try:
            all_chunks += extract_evidence(
                doc,
                use_llm=settings.evidence_llm_extract_enabled,
            )
        except Exception as exc:
            failures.append(
                _failure(
                    source_type=doc.source_type,
                    source_url=doc.source_url,
                    stage="extract",
                    exc=exc,
                )
            )

    try:
        all_chunks = chunk(all_chunks)
    except Exception as exc:
        failures.append(
            _failure(
                source_type="evidence",
                source_url=None,
                stage="chunk",
                exc=exc,
            )
        )
        all_chunks = []

    all_chunks = refine_evidence_chunks(all_chunks)

    store.add_chunks(all_chunks, user_id=user_id)
    coverage_map = store.build_coverage_map(user_id=user_id)

    return IndexBuildResult(
        status=_build_status(chunk_count=len(all_chunks), failures=failures),
        coverage_map=coverage_map,
        raw_doc_count=len(raw_docs),
        chunk_count=len(all_chunks),
        failures=failures,
    )


def _failure(
    source_type: str,
    source_url: str | None,
    stage: str,
    exc: BaseException,
) -> IndexFailure:
    """예외를 API 응답에 안전하게 담을 수 있는 실패 정보로 변환한다."""
    return IndexFailure(
        source_type=source_type,
        source_url=source_url,
        stage=stage,
        message=_format_exception(exc),
    )


def _format_exception(exc: BaseException) -> str:
    """ExceptionGroup 내부 원인까지 펼쳐 인덱싱 실패 메시지로 변환한다."""
    if isinstance(exc, BaseExceptionGroup):
        return " | ".join(_format_exception(child) for child in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


def _build_status(chunk_count: int, failures: list[IndexFailure]) -> str:
    """저장된 청크 수와 실패 목록을 기준으로 인덱싱 상태를 결정한다."""
    if chunk_count > 0 and failures:
        return "partial_failed"
    if chunk_count > 0:
        return "success"
    return "failed"
