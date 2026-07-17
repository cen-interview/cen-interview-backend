"""Evidence indexing 파이프라인의 링크 목록 계약을 검증한다."""

from interview.evidence.indexing import build_index
from interview.schemas.evidence import CoverageMap, EvidenceChunk, IndexBuildResult
from interview.evidence.sources import RawDoc


def test_build_index_passes_notion_and_github_link_lists(monkeypatch) -> None:
    """build_index가 Notion/GitHub 링크 목록을 source 계층에 그대로 전달해야 한다."""
    notion_links = ["https://notion.so/study-1", "https://notion.so/study-2"]
    github_links = [
        "https://github.com/example/project-1",
        "https://github.com/example/project-2",
        "https://github.com/example/project-3",
        "https://github.com/example/project-4",
    ]
    calls: dict[str, object] = {}

    class FakeNotionSource:
        """build_index가 전달한 Notion 링크 호출을 기록하는 fake source."""

        def fetch_pages(self, links: list[str]) -> list[object]:
            """Notion 링크별 호출을 기록하고 원본 문서가 없는 상태를 흉내낸다."""
            calls.setdefault("notion_link_calls", []).append(links)
            return []

    class FakeGitHubSource:
        """build_index가 전달한 GitHub 링크 호출을 기록하는 fake source."""

        def fetch_repos(
            self,
            links: list[str],
            github_login: str | None = None,
            github_verified_emails: list[str] | None = None,
        ) -> list[object]:
            """GitHub 링크별 호출을 기록하고 원본 문서가 없는 상태를 흉내낸다."""
            calls.setdefault("github_link_calls", []).append(links)
            calls.setdefault("github_login_calls", []).append(github_login)
            calls.setdefault("github_email_calls", []).append(github_verified_emails)
            return []

    class FakeStore:
        """build_index가 store에 넘기는 청크와 user_id를 기록하는 fake store."""

        def clear_user(self, user_id: str | None = None) -> None:
            """재인덱싱 전 초기화에 사용된 사용자 ID를 기록한다."""
            calls["clear_user_id"] = user_id

        def add_chunks(self, chunks: list[object], user_id: str | None = None) -> None:
            """저장 대상 청크와 사용자 ID를 기록한다."""
            calls["chunks"] = chunks
            calls["user_id"] = user_id

        def build_coverage_map(self, user_id: str | None = None) -> CoverageMap:
            """커버리지 생성에 사용된 사용자 ID를 기록하고 빈 CoverageMap을 반환한다."""
            calls["coverage_user_id"] = user_id
            return CoverageMap()

    monkeypatch.setattr("interview.evidence.indexing.NotionSource", FakeNotionSource)
    monkeypatch.setattr("interview.evidence.indexing.GitHubSource", FakeGitHubSource)
    monkeypatch.setattr("interview.evidence.indexing.chunk", lambda chunks: chunks)
    monkeypatch.setattr("interview.evidence.indexing.get_store", lambda: FakeStore())

    result = build_index(
        notion_links,
        github_links,
        user_id="user-1",
        github_login="octocat",
        github_verified_emails=["octocat@example.com"],
    )

    assert isinstance(result, IndexBuildResult)
    assert isinstance(result.coverage_map, CoverageMap)
    assert result.status == "failed"
    assert result.raw_doc_count == 0
    assert result.chunk_count == 0
    assert result.failures == []
    assert calls["clear_user_id"] == "user-1"
    assert calls["notion_link_calls"] == [[link] for link in notion_links]
    assert calls["github_link_calls"] == [[link] for link in github_links]
    assert calls["github_login_calls"] == ["octocat"] * len(github_links)
    assert calls["github_email_calls"] == [["octocat@example.com"]] * len(github_links)
    assert calls["chunks"] == []
    assert calls["user_id"] == "user-1"
    assert calls["coverage_user_id"] == "user-1"


def test_build_index_records_partial_failures_and_keeps_successful_chunks(monkeypatch) -> None:
    """일부 source가 실패해도 성공한 문서는 청크로 저장하고 실패 정보를 반환한다."""
    calls: dict[str, object] = {}

    raw_doc = RawDoc(
        source_url="https://notion.so/study",
        source_type="notion",
        title="study",
        raw_text="Spring Security JWT 인증 문제를 해결한 구체적인 회고입니다.",
        meta={"topic": "Spring Security", "doc_type": "retrospective"},
    )
    chunk = EvidenceChunk(
        chunk_id="chunk-1",
        text="Spring Security JWT 인증 문제를 해결한 구체적인 회고입니다.",
        source_type="notion",
        source_url=raw_doc.source_url,
        topic="spring security",
        doc_type="retrospective",
        week=None,
        date=None,
        confidence=0.8,
    )

    class FakeNotionSource:
        """첫 Notion 링크는 성공하고 두 번째 링크는 실패시키는 fake source."""

        def fetch_pages(self, links: list[str]) -> list[RawDoc]:
            """링크별 부분 실패를 검증하기 위해 단일 링크 호출을 처리한다."""
            if "bad" in links[0]:
                raise RuntimeError("notion unavailable")
            return [raw_doc]

    class FakeGitHubSource:
        """GitHub 링크는 실패시키는 fake source."""

        def fetch_repos(
            self,
            links: list[str],
            github_login: str | None = None,
            github_verified_emails: list[str] | None = None,
        ) -> list[RawDoc]:
            """GitHub source 장애를 흉내낸다."""
            _ = (github_login, github_verified_emails)
            raise RuntimeError("github unavailable")

    class FakeStore:
        """저장된 청크와 사용자 ID를 기록하는 fake store."""

        def clear_user(self, user_id: str | None = None) -> None:
            """재인덱싱 전 초기화에 사용된 사용자 ID를 기록한다."""
            calls["clear_user_id"] = user_id

        def add_chunks(self, chunks: list[EvidenceChunk], user_id: str | None = None) -> None:
            """성공한 청크가 실패 source와 무관하게 저장되는지 기록한다."""
            calls["chunks"] = chunks
            calls["user_id"] = user_id

        def build_coverage_map(self, user_id: str | None = None) -> CoverageMap:
            """성공 청크 기준의 CoverageMap을 반환한다."""
            calls["coverage_user_id"] = user_id
            return CoverageMap(
                topic_coverage={
                    "spring security": {"confidence": 0.8, "chunk_count": 1}
                }
            )

    monkeypatch.setattr("interview.evidence.indexing.NotionSource", FakeNotionSource)
    monkeypatch.setattr("interview.evidence.indexing.GitHubSource", FakeGitHubSource)
    monkeypatch.setattr(
        "interview.evidence.indexing.extract_evidence",
        lambda doc, use_llm=False: [chunk],
    )
    monkeypatch.setattr("interview.evidence.indexing.chunk", lambda chunks: chunks)
    monkeypatch.setattr("interview.evidence.indexing.get_store", lambda: FakeStore())

    result = build_index(
        ["https://notion.so/study", "https://notion.so/bad"],
        ["https://github.com/example/bad"],
        user_id="user-1",
    )

    assert result.status == "partial_failed"
    assert result.raw_doc_count == 1
    assert result.chunk_count == 1
    assert len(result.failures) == 2
    assert {failure.source_type for failure in result.failures} == {"notion", "github"}
    assert [item.chunk_id for item in calls["chunks"]] == ["chunk-1"]
    assert calls["chunks"][0].topic == "spring security"
    assert calls["user_id"] == "user-1"
