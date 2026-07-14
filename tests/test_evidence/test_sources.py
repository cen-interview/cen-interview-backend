"""Notion/GitHub source가 등록된 링크 목록을 모두 순회하는지 검증한다."""

import pytest

from interview.evidence.sources import (
    GitHubSource,
    NotionSource,
    RawDoc,
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
    ) -> dict:
        """GitHub MCP 호출 요청 링크를 기록하고 변환 가능한 응답을 돌려준다."""
        self.github_calls.append(repo_link)
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
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "commit abcdefabcdefabcdefabcdefabcdefabcdefabcd\n"
                                "modified src/service.py"
                            ),
                        }
                    ]
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


def test_github_response_to_raw_docs_creates_readme_meta_and_tree_docs() -> None:
    """GitHub MCP 응답에서 README, 저장소 메타, 디렉터리 트리 RawDoc을 만든다."""
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

    assert [doc.meta["doc_type"] for doc in docs] == [
        "README",
        "repository_meta",
        "directory_tree",
    ]
    assert docs[0].raw_text.startswith("# Project")
    assert docs[0].meta["file_path"] == "README.md"
    assert docs[1].raw_text == "language: Python"
    assert "src/app.py" in docs[2].raw_text


def test_github_response_to_raw_docs_skips_missing_readme() -> None:
    """README가 없어도 저장소 메타와 디렉터리 트리는 RawDoc으로 만든다."""
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

    assert [doc.meta["doc_type"] for doc in docs] == [
        "repository_meta",
        "directory_tree",
    ]


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
    """선별된 코드 파일 RawDoc에 사용자 기여 여부 meta를 표시한다."""
    client = FakeMcpClient()

    docs = GitHubSource(mcp_client=client).fetch_repos(
        ["https://github.com/example/project"],
        github_login="octocat",
    )

    code_docs = [doc for doc in docs if doc.meta.get("doc_type") == "code"]

    assert [doc.meta["file_path"] for doc in code_docs] == [
        "src/service.py",
        "src/main.py",
    ]
    assert code_docs[0].meta["ownership"] == "user_touched"
    assert code_docs[0].meta["author_login"] == "octocat"
    assert code_docs[0].meta["commit_count"] == 1
    assert code_docs[0].raw_text.startswith("// contents for example/project/src/service.py")
    assert code_docs[1].meta["ownership"] == "repo_context"
    assert "tests/test_service.py" not in [doc.meta["file_path"] for doc in code_docs]


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
