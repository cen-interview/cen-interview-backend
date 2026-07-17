"""Notion/GitHub source가 등록된 링크 목록을 모두 순회하는지 검증한다."""

from contextlib import asynccontextmanager

import anyio
import pytest

from interview.evidence.mcp_client import EvidenceMcpClient, _extract_github_commit_shas
from interview.evidence.sources import (
    GitHubSource,
    NotionSource,
    RawDoc,
    _extract_added_patch_hunks,
    _extract_github_file_paths,
    _normalize_notion_link,
    _parse_github_repo_url,
    _select_github_code_paths,
)


class FakeMcpClient:
    """실제 MCP 호출 없이 source의 순회 동작만 검증하기 위한 fake client."""

    def __init__(self) -> None:
        self.notion_calls: list[str] = []
        self.github_calls: list[str] = []

    def call_notion_tool(
        self,
        root_link: str,
        tool_name: str,
        arguments: dict | None = None,
    ) -> dict:
        """Notion MCP 호출 요청 링크를 기록하고 변환 가능한 응답을 돌려준다."""
        self.notion_calls.append(root_link)
        return {"url": root_link, "title": f"notion:{root_link}"}

    def call_github_tool(
        self,
        repo_link: str,
        owner: str | None = None,
        repo: str | None = None,
        github_login: str | None = None,
        github_verified_emails: list[str] | None = None,
    ) -> dict:
        """GitHub MCP 호출 요청 링크를 기록하고 변환 가능한 응답을 돌려준다."""
        self.github_calls.append(repo_link)
        _ = (github_login, github_verified_emails)
        repo_name = f"{owner}/{repo}" if owner and repo else repo_link
        return {
            "repo_url": repo_link,
            "owner": owner,
            "repo": repo,
            "title": f"github:{repo_link}",
            "repository": {
                "content": [{"type": "text", "text": f"metadata for {repo_name}"}]
            },
            "readme": {
                "content": [{"type": "text", "text": f"# README for {repo_name}"}]
            },
            "tree": {
                "content": [
                    {
                        "type": "text",
                        "text": "\n".join(
                            [
                                "src/main.py",
                                "src/service.py",
                                "tests/test_service.py",
                                "package-lock.json",
                                "README.md",
                            ]
                        ),
                    }
                ]
            },
            "commit_details": [
                {
                    "structuredContent": {
                        "sha": "a" * 40,
                        "author": {"login": "octocat"},
                        "parents": [{"sha": "b" * 40}],
                        "files": [
                            {
                                "filename": "src/service.py",
                                "status": "modified",
                                "patch": (
                                    "@@ -1,2 +1,4 @@\n"
                                    "-def send():\n"
                                    "+def send_message(message):\n"
                                    "+    websocket.send(message)\n"
                                    "+    return message\n"
                                    " context_line()"
                                ),
                            }
                        ],
                    }
                }
            ],
        }

    def fetch_github_file_contents(
        self,
        owner: str,
        repo: str,
        paths: list[str],
    ) -> dict[str, dict]:
        """선별된 GitHub 파일 경로에 대한 MCP contents 응답을 돌려준다."""
        return {
            path: {
                "content": [
                    {
                        "type": "text",
                        "text": "successfully downloaded text file (SHA: test)",
                    },
                    {
                        "type": "resource",
                        "resource": {
                            "mimeType": "text/plain",
                            "text": f"// contents for {owner}/{repo}/{path}",
                            "uri": f"repo://{owner}/{repo}/{path}",
                        },
                    },
                ]
            }
            for path in paths
        }


def test_github_file_contents_respect_concurrency_limit(monkeypatch) -> None:
    """파일 본문 조회는 설정된 한도 안에서 병렬 실행하고 입력 순서를 보존한다."""
    active = 0
    peak = 0

    class FakeResult:
        def __init__(self, path: str) -> None:
            self.path = path

        def model_dump(self, mode: str) -> dict:
            _ = mode
            return {"path": self.path}

    class FakeSession:
        async def call_tool(self, tool_name: str, arguments: dict) -> FakeResult:
            nonlocal active, peak
            _ = tool_name
            active += 1
            peak = max(peak, active)
            await anyio.sleep(0.01)
            active -= 1
            return FakeResult(arguments["path"])

    session = FakeSession()

    @asynccontextmanager
    async def fake_github_session():
        yield session

    monkeypatch.setattr(
        "interview.evidence.mcp_client.settings.evidence_mcp_concurrency",
        2,
    )
    client = EvidenceMcpClient(
        github_mcp_url="https://example.com/mcp",
        github_access_token="token",
    )
    monkeypatch.setattr(client, "_github_session", fake_github_session)
    paths = ["src/a.py", "src/b.py", "src/c.py", "src/d.py"]

    result = client.fetch_github_file_contents("owner", "repo", paths)

    assert list(result) == paths
    assert peak == 2


def test_github_commit_details_request_full_patch() -> None:
    """사용자 실제 변경 코드를 얻기 위해 commit detail은 full_patch를 요청한다."""
    calls: list[tuple[str, dict]] = []

    class FakeResult:
        def model_dump(self, mode: str) -> dict:
            _ = mode
            return {"structuredContent": {"sha": "a" * 40}}

    class FakeSession:
        async def call_tool(self, tool_name: str, arguments: dict) -> FakeResult:
            calls.append((tool_name, arguments))
            return FakeResult()

    client = EvidenceMcpClient(
        github_mcp_url="https://example.com/mcp",
        github_access_token="token",
    )

    async def fetch() -> list[dict]:
        return await client._fetch_github_commit_details(
            FakeSession(),
            "owner",
            "repo",
            ["a" * 40],
        )

    result = anyio.run(fetch)

    assert result
    assert calls[0][1]["detail"] == "full_patch"


def test_github_commit_lookup_uses_verified_email_before_login(monkeypatch) -> None:
    """이메일 연결이 안 된 commit도 찾도록 검증 이메일을 login보다 먼저 조회한다."""
    calls: list[dict] = []
    first_sha = "a" * 40
    second_sha = "b" * 40
    third_sha = "c" * 40
    client = EvidenceMcpClient(
        github_mcp_url="https://example.com/mcp",
        github_access_token="token",
    )

    async def fake_call(session, tool_name: str, arguments: dict) -> dict:
        _ = (session, tool_name)
        calls.append(arguments)
        author = arguments["author"]
        page = arguments["page"]
        if author == "octocat@example.com":
            shas = [first_sha]
        elif page == 1:
            shas = [first_sha, second_sha]
        else:
            shas = [third_sha]
        return {"structuredContent": [{"sha": sha} for sha in shas]}

    monkeypatch.setattr(client, "_safe_call_github_tool", fake_call)

    async def fetch() -> tuple[dict, list[str]]:
        return await client._fetch_github_commits(
            session=object(),
            owner="example",
            repo="project",
            github_login="octocat",
            github_verified_emails=["octocat@example.com"],
            max_commits=3,
        )

    _, shas = anyio.run(fetch)

    assert [call["author"] for call in calls] == [
        "octocat@example.com",
        "octocat",
        "octocat",
    ]
    assert shas == [first_sha, second_sha, third_sha]


def test_github_mcp_retries_nested_rate_limit_errors(monkeypatch) -> None:
    """TaskGroup 내부 429도 설정된 백오프를 적용해 재시도한다."""
    attempts = 0
    delays: list[float] = []

    async def rate_limited_then_success() -> dict:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ExceptionGroup("mcp", [RuntimeError("429 Too Many Requests")])
        return {"ok": True}

    monkeypatch.setattr(
        "interview.evidence.mcp_client.settings.evidence_mcp_max_attempts",
        3,
    )
    monkeypatch.setattr(
        "interview.evidence.mcp_client.settings.evidence_mcp_retry_base_seconds",
        0.5,
    )
    monkeypatch.setattr("interview.evidence.mcp_client.time.sleep", delays.append)
    client = EvidenceMcpClient(
        github_mcp_url="https://example.com/mcp",
        github_access_token="token",
    )

    result = client._run_github_with_retry(rate_limited_then_success)

    assert result == {"ok": True}
    assert attempts == 3
    assert delays == [0.5, 1.0]


def test_extract_github_commit_shas_ignores_parent_and_tree_shas() -> None:
    """list 응답에서는 commit 레코드의 sha만 수집해야 한다."""
    commit_sha = "a" * 40
    parent_sha = "b" * 40
    tree_sha = "c" * 40
    response = {
        "structuredContent": {
            "commits": [
                {
                    "sha": commit_sha,
                    "parents": [{"sha": parent_sha}],
                    "commit": {"tree": {"sha": tree_sha}},
                }
            ]
        }
    }

    assert _extract_github_commit_shas(response) == [commit_sha]


def test_extract_added_patch_hunks_separates_changes_across_context() -> None:
    """같은 diff hunk에서도 unchanged context로 떨어진 추가 코드는 분리한다."""
    contributions = _extract_added_patch_hunks(
        "a" * 40,
        "src/service.py",
        (
            "@@ -10,4 +10,6 @@\n"
            "+first_added()\n"
            " existing_line()\n"
            "+second_added()"
        ),
    )

    assert [item.text for item in contributions] == ["first_added()", "second_added()"]
    assert [(item.start_line, item.end_line) for item in contributions] == [
        (10, 10),
        (12, 12),
    ]


def test_notion_source_fetches_all_registered_links(monkeypatch) -> None:
    """NotionSource가 단일 링크가 아니라 등록된 링크 전체를 순회해야 한다."""
    client = FakeMcpClient()
    links = [
        "https://notion.so/study-1",
        "https://notion.so/study-2",
        "https://notion.so/study-3",
    ]

    def fake_convert(response: dict, root_link: str) -> list[RawDoc]:
        """Notion MCP 응답 1건을 테스트용 RawDoc 1건으로 바꾼다."""
        return [
            RawDoc(
                source_url=root_link,
                source_type="notion",
                title=response["title"],
                raw_text=f"{root_link} raw text",
                meta={},
            )
        ]

    monkeypatch.setattr("interview.evidence.sources._notion_response_to_raw_docs", fake_convert)

    docs = NotionSource(mcp_client=client).fetch_pages(links)

    assert client.notion_calls == links
    assert [doc.source_url for doc in docs] == links


def test_notion_source_skips_mcp_error_response() -> None:
    """notion-fetch가 isError를 반환하면 에러 문구를 RawDoc으로 만들지 않는다."""

    class ErrorMcpClient:
        def call_notion_tool(
            self,
            root_link: str,
            tool_name: str,
            arguments: dict | None = None,
        ) -> dict:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "MCP error"}],
            }

    docs = NotionSource(mcp_client=ErrorMcpClient()).fetch_pages(
        ["https://app.notion.com/p/36b80110f2e88027a17fc21ace0da8e8"]
    )

    assert docs == []


def test_normalize_notion_link_canonicalizes_app_page_slug() -> None:
    """Notion app page URL의 slug prefix를 canonical page ID URL로 정리한다."""
    link = "https://app.notion.com/p/_5-4-36b80110f2e88027a17fc21ace0da8e8"

    assert (
        _normalize_notion_link(link)
        == "https://app.notion.com/p/36b80110f2e88027a17fc21ace0da8e8"
    )


def test_github_source_fetches_all_registered_links_without_limit(monkeypatch) -> None:
    """GitHubSource가 링크를 3개로 자르지 않고 등록된 목록 전체를 순회해야 한다."""
    client = FakeMcpClient()
    links = [
        "https://github.com/example/project-1",
        "https://github.com/example/project-2",
        "https://github.com/example/project-3",
        "https://github.com/example/project-4",
    ]

    def fake_convert(response: dict, repo_link: str) -> list[RawDoc]:
        """GitHub MCP 응답 1건을 테스트용 RawDoc 1건으로 바꾼다."""
        return [
            RawDoc(
                source_url=repo_link,
                source_type="github",
                title=response["title"],
                raw_text=f"{repo_link} raw text",
                meta={},
            )
        ]

    monkeypatch.setattr("interview.evidence.sources._github_response_to_raw_docs", fake_convert)

    docs = GitHubSource(mcp_client=client).fetch_repos(links)

    base_docs = [doc for doc in docs if doc.meta == {}]

    assert client.github_calls == links
    assert [doc.source_url for doc in base_docs] == links


def test_parse_github_repo_url_supports_https_url() -> None:
    """HTTPS GitHub repository URL에서 owner/repo를 추출한다."""
    assert _parse_github_repo_url("https://github.com/openai/openai-python") == (
        "openai",
        "openai-python",
    )


def test_parse_github_repo_url_strips_git_suffix() -> None:
    """.git suffix가 붙은 repository URL도 같은 owner/repo로 정규화한다."""
    assert _parse_github_repo_url("https://github.com/openai/openai-python.git") == (
        "openai",
        "openai-python",
    )


def test_parse_github_repo_url_supports_ssh_url() -> None:
    """SSH 형식의 GitHub repository URL도 파싱한다."""
    assert _parse_github_repo_url("git@github.com:openai/openai-python.git") == (
        "openai",
        "openai-python",
    )


def test_parse_github_repo_url_rejects_non_github_url() -> None:
    """GitHub이 아닌 URL은 repository 링크로 보지 않는다."""
    with pytest.raises(ValueError):
        _parse_github_repo_url("https://gitlab.com/openai/openai-python")


def test_github_source_skips_invalid_repo_links(monkeypatch) -> None:
    """잘못된 GitHub 링크는 전체 수집을 중단하지 않고 스킵한다."""
    client = FakeMcpClient()
    links = [
        "https://github.com/example/project",
        "https://gitlab.com/example/project",
        "not-a-url",
    ]

    def fake_convert(response: dict, repo_link: str) -> list[RawDoc]:
        """GitHub MCP 응답 1건을 테스트용 RawDoc 1건으로 바꾼다."""
        return [
            RawDoc(
                source_url=repo_link,
                source_type="github",
                title=response["title"],
                raw_text=f"{repo_link} raw text",
                meta={},
            )
        ]

    monkeypatch.setattr("interview.evidence.sources._github_response_to_raw_docs", fake_convert)

    docs = GitHubSource(mcp_client=client).fetch_repos(links)

    base_docs = [doc for doc in docs if doc.meta == {}]

    assert client.github_calls == ["https://github.com/example/project"]
    assert [doc.source_url for doc in base_docs] == ["https://github.com/example/project"]


def test_github_response_to_raw_docs_keeps_readme_but_excludes_meta_and_tree() -> None:
    """저장소 메타와 트리는 임베딩하지 않고 README만 RawDoc으로 만든다."""
    response = {
        "repo_url": "https://github.com/example/project",
        "owner": "example",
        "repo": "project",
        "repository": {"content": [{"type": "text", "text": "language: Python"}]},
        "readme": {"content": [{"type": "text", "text": "# Project\n\nREADME body"}]},
        "tree": {"content": [{"type": "text", "text": "src/app.py\nREADME.md"}]},
    }

    from interview.evidence.sources import _github_response_to_raw_docs

    docs = _github_response_to_raw_docs(response, "https://github.com/example/project")

    assert [doc.meta["doc_type"] for doc in docs] == ["README"]
    assert docs[0].raw_text.startswith("# Project")
    assert docs[0].meta["file_path"] == "README.md"


def test_github_response_to_raw_docs_skips_missing_readme() -> None:
    """README가 없으면 메타와 디렉터리 트리도 임베딩 대상으로 만들지 않는다."""
    response = {
        "repo_url": "https://github.com/example/project",
        "owner": "example",
        "repo": "project",
        "repository": {"content": [{"type": "text", "text": "metadata"}]},
        "readme": {"isError": True, "content": [{"type": "text", "text": "not found"}]},
        "tree": {"content": [{"type": "text", "text": "src/app.py"}]},
    }

    from interview.evidence.sources import _github_response_to_raw_docs

    docs = _github_response_to_raw_docs(response, "https://github.com/example/project")

    assert docs == []


def test_github_response_to_raw_docs_skips_download_status_readme() -> None:
    """README 다운로드 상태 메시지는 실제 README 본문으로 저장하지 않는다."""
    response = {
        "repo_url": "https://github.com/example/project",
        "owner": "example",
        "repo": "project",
        "repository": {"content": [{"type": "text", "text": "metadata"}]},
        "readme": {
            "content": [
                {
                    "type": "text",
                    "text": "successfully downloaded text file (SHA: abcdef)",
                }
            ]
        },
        "tree": {"content": [{"type": "text", "text": "src/app.py"}]},
    }

    from interview.evidence.sources import _github_response_to_raw_docs

    docs = _github_response_to_raw_docs(response, "https://github.com/example/project")

    assert "README" not in [doc.meta["doc_type"] for doc in docs]


def test_github_source_creates_code_docs_with_ownership() -> None:
    """현재 파일은 context로, 검증된 추가 코드는 별도 user_touched로 만든다."""
    client = FakeMcpClient()

    docs = GitHubSource(mcp_client=client).fetch_repos(
        ["https://github.com/example/project"],
        github_login="octocat",
    )

    code_docs = [doc for doc in docs if doc.meta.get("doc_type") == "code"]
    context_docs = [doc for doc in code_docs if doc.meta["ownership"] == "repo_context"]
    contribution_docs = [doc for doc in code_docs if doc.meta["ownership"] == "user_touched"]
    context_by_path = {doc.meta["file_path"]: doc for doc in context_docs}

    assert set(context_by_path) == {
        "src/service.py",
        "src/main.py",
        "tests/test_service.py",
    }
    assert context_by_path["src/service.py"].raw_text.startswith(
        "// contents for example/project/src/service.py"
    )
    assert len(contribution_docs) == 1
    contribution = contribution_docs[0]
    assert contribution.meta["file_path"] == "src/service.py"
    assert contribution.meta["author_login"] == "octocat"
    assert contribution.meta["commit_count"] == 1
    assert contribution.meta["last_commit_sha"] == "a" * 40
    assert contribution.raw_text == (
        "def send_message(message):\n"
        "    websocket.send(message)\n"
        "    return message"
    )
    assert "/blob/" + "a" * 40 in contribution.source_url
    assert contribution.source_url.endswith("#L1-L3")


def test_github_source_excludes_other_author_and_merge_contributions() -> None:
    """다른 작성자와 merge commit은 user_touched 근거로 저장하지 않는다."""

    class UnverifiedCommitClient(FakeMcpClient):
        def call_github_tool(self, *args, **kwargs) -> dict:
            response = super().call_github_tool(*args, **kwargs)
            base_file = {
                "filename": "src/service.py",
                "status": "modified",
                "patch": "@@ -1 +1 @@\n-old\n+def changed_by_commit(): return True",
            }
            response["commit_details"] = [
                {
                    "structuredContent": {
                        "sha": "c" * 40,
                        "author": {"login": "someone-else"},
                        "parents": [{"sha": "d" * 40}],
                        "files": [base_file],
                    }
                },
                {
                    "structuredContent": {
                        "sha": "e" * 40,
                        "author": {"login": "octocat"},
                        "parents": [{"sha": "f" * 40}, {"sha": "1" * 40}],
                        "files": [base_file],
                    }
                },
            ]
            return response

    docs = GitHubSource(mcp_client=UnverifiedCommitClient()).fetch_repos(
        ["https://github.com/example/project"],
        github_login="octocat",
    )

    assert not [doc for doc in docs if doc.meta.get("ownership") == "user_touched"]
    assert [doc for doc in docs if doc.meta.get("ownership") == "repo_context"]


def test_github_source_accepts_verified_commit_when_mcp_omits_parents() -> None:
    """실제 GitHub MCP처럼 parents가 없어도 작성자와 patch가 검증되면 저장한다."""

    class ParentlessCommitClient(FakeMcpClient):
        def call_github_tool(self, *args, **kwargs) -> dict:
            response = super().call_github_tool(*args, **kwargs)
            record = response["commit_details"][0]["structuredContent"]
            record.pop("parents")
            record["commit"] = {"message": "feat: implement websocket message send"}
            return response

    docs = GitHubSource(mcp_client=ParentlessCommitClient()).fetch_repos(
        ["https://github.com/example/project"],
        github_login="octocat",
    )

    assert [doc for doc in docs if doc.meta.get("ownership") == "user_touched"]


def test_github_source_accepts_verified_commit_email_without_linked_login() -> None:
    """GitHub login 연결이 누락돼도 검증 이메일이 같은 일반 commit은 저장한다."""

    class EmailMatchedCommitClient(FakeMcpClient):
        def call_github_tool(self, *args, **kwargs) -> dict:
            response = super().call_github_tool(*args, **kwargs)
            record = response["commit_details"][0]["structuredContent"]
            record["author"] = None
            record["commit"] = {
                "author": {"name": "octocat", "email": "octocat@example.com"},
                "message": "feat: implement websocket message send",
            }
            return response

    docs = GitHubSource(mcp_client=EmailMatchedCommitClient()).fetch_repos(
        ["https://github.com/example/project"],
        github_login="octocat",
        github_verified_emails=["OctoCat@example.com"],
    )

    assert [doc for doc in docs if doc.meta.get("ownership") == "user_touched"]


def test_github_source_rejects_unverified_commit_email() -> None:
    """API login과 검증 이메일이 모두 불일치하면 user_touched로 저장하지 않는다."""

    class EmailMismatchCommitClient(FakeMcpClient):
        def call_github_tool(self, *args, **kwargs) -> dict:
            response = super().call_github_tool(*args, **kwargs)
            record = response["commit_details"][0]["structuredContent"]
            record["author"] = None
            record["commit"] = {
                "author": {"name": "octocat", "email": "unknown@example.com"},
                "message": "feat: implement websocket message send",
            }
            return response

    docs = GitHubSource(mcp_client=EmailMismatchCommitClient()).fetch_repos(
        ["https://github.com/example/project"],
        github_login="octocat",
        github_verified_emails=["octocat@example.com"],
    )

    assert not [doc for doc in docs if doc.meta.get("ownership") == "user_touched"]
    assert [doc for doc in docs if doc.meta.get("ownership") == "repo_context"]


def test_github_source_excludes_merge_message_when_mcp_omits_parents() -> None:
    """parents가 없는 응답은 merge commit 메시지로 병합 기록을 제외한다."""

    class ParentlessMergeClient(FakeMcpClient):
        def call_github_tool(self, *args, **kwargs) -> dict:
            response = super().call_github_tool(*args, **kwargs)
            record = response["commit_details"][0]["structuredContent"]
            record.pop("parents")
            record["commit"] = {"message": "Merge pull request #10 from feature/chat"}
            return response

    docs = GitHubSource(mcp_client=ParentlessMergeClient()).fetch_repos(
        ["https://github.com/example/project"],
        github_login="octocat",
    )

    assert not [doc for doc in docs if doc.meta.get("ownership") == "user_touched"]


def test_extract_github_file_paths_restores_directory_context() -> None:
    """재귀 directory 응답에서 상대 파일명을 parent directory path와 합친다."""
    tree_response = {
        "content": [
            {
                "type": "text",
                "text": "README.md\n\n# src\nmain.py\nservice.py",
            }
        ]
    }

    assert _extract_github_file_paths(tree_response) == [
        "README.md",
        "src/main.py",
        "src/service.py",
    ]


def test_select_github_code_paths_prioritizes_main_source_over_resources() -> None:
    """핵심 구현 파일을 html/sql resource보다 먼저 선별한다."""
    paths = [
        "bugbug/chat-test.html",
        "bugbug/src/main/resources/data.sql",
        "bugbug/src/main/java/com/example/bugbug/user/UserController.java",
        "bugbug/src/main/java/com/example/bugbug/user/UserService.java",
    ]

    assert _select_github_code_paths(paths, touched_files={}, max_files=3) == [
        "bugbug/src/main/java/com/example/bugbug/user/UserController.java",
        "bugbug/src/main/java/com/example/bugbug/user/UserService.java",
        "bugbug/chat-test.html",
    ]


def test_select_github_code_paths_excludes_seed_and_dummy_sql() -> None:
    """면접 근거 가치가 낮은 seed/dummy SQL 파일은 코드 수집 후보에서 제외한다."""
    paths = [
        "bugbug/src/main/resources/data.sql",
        "bugbug/src/main/resources/dummy-users.sql",
        "bugbug/src/main/resources/seed-users.sql",
        "bugbug/src/main/java/com/example/bugbug/user/UserService.java",
    ]

    assert _select_github_code_paths(paths, touched_files={}, max_files=10) == [
        "bugbug/src/main/java/com/example/bugbug/user/UserService.java",
    ]


def test_select_github_code_paths_uses_zero_as_unlimited() -> None:
    """파일 제한이 0이면 테스트를 포함한 모든 유효 소스를 반환한다."""
    paths = [
        "src/main.py",
        "src/pipeline.py",
        "src/model.py",
        "tests/test_pipeline.py",
        "node_modules/library/index.js",
    ]
    touched = {"src/pipeline.py": ["a" * 40]}

    selected = _select_github_code_paths(paths, touched_files=touched, max_files=0)

    assert selected[0] == "src/pipeline.py"
    assert set(selected) == {
        "src/pipeline.py",
        "src/main.py",
        "src/model.py",
        "tests/test_pipeline.py",
    }
