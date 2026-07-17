import anyio
import httpx
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, TypeVar

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from interview.config import settings


T = TypeVar("T")


class EvidenceMcpClient:
    """Evidence 수집 과정에서 Notion/GitHub MCP tool call을 실행하는 얇은 래퍼.

    이 클래스는 MCP 연결과 tool 호출만 담당한다. 응답을 RawDoc으로 바꾸는 일은
    sources.py의 NotionSource/GitHubSource가 담당한다.
    """
    def __init__(
          self,
          notion_mcp_url: str | None = None,
          notion_access_token: str | None = None,
          github_mcp_url: str | None = None,
          github_access_token: str | None = None,
          timeout_seconds: float = 30.0,
      ) -> None:
          """MCP 호출에 필요한 endpoint와 인증 정보를 초기화한다.

          Args:
              notion_mcp_url: Notion MCP Streamable HTTP endpoint.
                  None이면 settings.notion_mcp_url을 사용한다.
              notion_access_token: Notion OAuth access token.
                  None이면 settings.notion_mcp_access_token을 사용한다.
                  서비스 요청에서는 저장된 사용자별 MCP token을 호출부에서 주입한다.
              github_mcp_url: GitHub MCP Streamable HTTP endpoint.
                  None이면 settings.github_mcp_url을 사용한다.
              github_access_token: GitHub MCP access token.
                  None이면 settings.github_mcp_access_token 또는 settings.github_token을 사용한다.
              timeout_seconds: MCP 연결과 tool call에 사용할 timeout 초 단위 값.
          """
          self.notion_mcp_url = notion_mcp_url or settings.notion_mcp_url
          self.notion_access_token = notion_access_token or settings.notion_mcp_access_token
          self.github_mcp_url = github_mcp_url or settings.github_mcp_url
          self.github_access_token = (
              github_access_token
              or settings.github_mcp_access_token
              or settings.github_token
          )
          self.timeout_seconds = timeout_seconds

    def list_notion_tools(self) -> list[dict]:
        """Notion MCP 서버가 제공하는 tool 목록과 입력 schema를 조회한다.

        실제 page/database 조회 tool 이름과 arguments 구조를 확정하기 전,
        개발자가 MCP 서버의 tool 계약을 확인할 수 있도록 사용한다.

        Returns:
            Notion MCP 서버가 노출하는 tool 목록. 각 항목에는 tool 이름,
            설명, input schema 등이 포함된다.

        Raises:
            ValueError:
                Notion MCP access token이 설정되어 있지 않은 경우.
        """
        if not self.notion_access_token:
            raise ValueError("Notion MCP 토큰이 필요합니다.")

        return anyio.run(self._list_notion_tools_async)
    
    def call_notion_tool(self, root_link: str, tool_name: str, arguments: dict | None = None) -> dict:
        """Notion MCP tool을 호출하고 원본 응답을 반환한다.

        Args:
            root_link: 사용자가 등록한 Notion page/database 링크.
            tool_name: list_notion_tools()로 확인한 실제 Notion 조회 tool 이름.
            arguments: tool 호출에 넘길 추가 arguments.
                None이면 root_link를 기본 argument로 전달한다.

        Returns:
            MCP tool call 원본 응답.
        """
        if not self.notion_access_token:
            raise ValueError("Notion MCP 토큰이 필요합니다.")

        tool_arguments = arguments or {"id": root_link}
        return anyio.run(self._call_notion_tool_async, tool_name, tool_arguments)

    @asynccontextmanager
    async def _notion_session(self) -> AsyncIterator[ClientSession]:
        """Notion MCP Streamable HTTP session을 열고 초기화한다.

        MCP 연결 생성, OAuth access token 주입, session.initialize() 호출을
        한곳에 모아 list_tools/call_tool 양쪽에서 같은 연결 절차를 재사용한다.
        """
        headers = {"Authorization": f"Bearer {self.notion_access_token}"}
        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(self.timeout_seconds),
        ) as http_client:
            async with streamable_http_client(
                self.notion_mcp_url,
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
                ) as session:
                    await session.initialize()
                    yield session

    async def _list_notion_tools_async(self) -> list[dict]:
        """초기화된 Notion MCP session으로 tool 목록을 비동기 조회한다."""
        async with self._notion_session() as session:
            tools = await session.list_tools()
            return [tool.model_dump(mode="json") for tool in tools.tools]


    async def _call_notion_tool_async(self, tool_name: str, arguments: dict) -> dict:
        """초기화된 Notion MCP session으로 지정 tool을 비동기 호출한다."""
        async with self._notion_session() as session:
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.model_dump(mode="json")

    def list_github_tools(self) -> list[dict]:
        """GitHub MCP 서버가 제공하는 tool 목록과 입력 schema를 조회한다."""
        if not self.github_mcp_url:
            raise ValueError("GitHub MCP URL이 필요합니다.")
        if not self.github_access_token:
            raise ValueError("GitHub MCP 토큰이 필요합니다.")

        return anyio.run(self._list_github_tools_async)

    def call_github_tool(
        self,
        repo_link: str,
        owner: str | None = None,
        repo: str | None = None,
        github_login: str | None = None,
        github_verified_emails: list[str] | None = None,
    ) -> dict:
        """GitHub MCP로 repository metadata, README, tree, commit 원본 응답을 모은다.

        공식 GitHub MCP에는 get_repository tool이 없으므로 repository
        metadata는 search_repositories로 조회한다. github_login이 있으면
        해당 사용자의 commit 목록도 함께 조회해 코드 RawDoc 메타에 반영한다.
        """
        if not self.github_mcp_url:
            raise ValueError("GitHub MCP URL이 필요합니다.")
        if not self.github_access_token:
            raise ValueError("GitHub MCP 토큰이 필요합니다.")
        if owner is None or repo is None:
            raise ValueError("GitHub owner/repo가 필요합니다.")

        return self._run_github_with_retry(
            self._call_github_repo_async,
            repo_link,
            owner,
            repo,
            github_login,
            github_verified_emails,
        )

    @asynccontextmanager
    async def _github_session(self) -> AsyncIterator[ClientSession]:
        """GitHub MCP Streamable HTTP session을 열고 초기화한다."""
        headers = {"Authorization": f"Bearer {self.github_access_token}"}
        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(self.timeout_seconds),
        ) as http_client:
            async with streamable_http_client(
                self.github_mcp_url,
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
                ) as session:
                    await session.initialize()
                    yield session

    async def _list_github_tools_async(self) -> list[dict]:
        """초기화된 GitHub MCP session으로 tool 목록을 비동기 조회한다."""
        async with self._github_session() as session:
            tools = await session.list_tools()
            return [tool.model_dump(mode="json") for tool in tools.tools]

    async def _call_github_tool_async(self, tool_name: str, arguments: dict) -> dict:
        """초기화된 GitHub MCP session으로 지정 tool을 비동기 호출한다."""
        async with self._github_session() as session:
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.model_dump(mode="json")

    async def _call_github_repo_async(
        self,
        repo_link: str,
        owner: str,
        repo: str,
        github_login: str | None,
        github_verified_emails: list[str] | None,
    ) -> dict:
        """GitHub repository, README, tree, commit MCP 응답을 한 번에 모은다."""
        async with self._github_session() as session:
            repository = await self._safe_call_github_tool(
                session,
                settings.github_mcp_repository_tool,
                {"query": f"repo:{owner}/{repo}", "perPage": 1},
            )
            readme = await self._safe_call_github_tool(
                session,
                settings.github_mcp_contents_tool,
                {"owner": owner, "repo": repo, "path": "README.md"},
            )
            tree = await self._fetch_github_directory_tree(
                session,
                owner,
                repo,
                max_dirs=settings.evidence_github_max_dirs,
                max_depth=settings.evidence_github_max_depth,
            )
            commits, commit_shas = await self._fetch_github_commits(
                session=session,
                owner=owner,
                repo=repo,
                github_login=github_login,
                github_verified_emails=github_verified_emails,
                max_commits=settings.evidence_github_max_commits,
            )
            commit_details = await self._fetch_github_commit_details(
                session,
                owner,
                repo,
                commit_shas,
            )

        return {
            "repo_url": repo_link,
            "owner": owner,
            "repo": repo,
            "repository": repository,
            "readme": readme,
            "tree": tree,
            "commits": commits,
            "commit_details": commit_details,
        }

    def fetch_github_file_contents(self, owner: str, repo: str, paths: list[str]) -> dict[str, dict]:
        """GitHub MCP로 지정 파일들의 원본 contents 응답을 가져온다."""
        if not self.github_mcp_url:
            raise ValueError("GitHub MCP URL이 필요합니다.")
        if not self.github_access_token:
            raise ValueError("GitHub MCP 토큰이 필요합니다.")

        return self._run_github_with_retry(
            self._fetch_github_file_contents_async,
            owner,
            repo,
            paths,
        )

    def _run_github_with_retry(
        self,
        async_call: Callable[..., Awaitable[T]],
        *args: Any,
    ) -> T:
        """GitHub MCP 429 응답을 지수 백오프로 재시도한다."""
        max_attempts = max(1, settings.evidence_mcp_max_attempts)
        for attempt in range(max_attempts):
            try:
                return anyio.run(async_call, *args)
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt == max_attempts - 1:
                    raise
                time.sleep(
                    max(0.0, settings.evidence_mcp_retry_base_seconds)
                    * (2**attempt)
                )

        raise RuntimeError("GitHub MCP 재시도 상태가 올바르지 않습니다.")

    async def _fetch_github_file_contents_async(
        self,
        owner: str,
        repo: str,
        paths: list[str],
    ) -> dict[str, dict]:
        """초기화된 GitHub MCP session으로 지정 파일 내용을 비동기 조회한다."""
        async with self._github_session() as session:
            responses: list[dict | None] = [None] * len(paths)
            limiter = anyio.Semaphore(max(1, settings.evidence_mcp_concurrency))

            async def fetch_one(index: int, path: str) -> None:
                async with limiter:
                    responses[index] = await self._safe_call_github_tool(
                        session,
                        settings.github_mcp_contents_tool,
                        {"owner": owner, "repo": repo, "path": path},
                    )

            async with anyio.create_task_group() as task_group:
                for index, path in enumerate(paths):
                    task_group.start_soon(fetch_one, index, path)

            return {
                path: response
                for path, response in zip(paths, responses, strict=True)
                if response is not None
            }

    async def _fetch_github_commit_details(
        self,
        session: ClientSession,
        owner: str,
        repo: str,
        commit_shas: list[str],
    ) -> list[dict]:
        """list_commits 응답의 sha들을 get_commit detail 응답 목록으로 확장한다."""
        details: list[dict | None] = [None] * len(commit_shas)
        limiter = anyio.Semaphore(max(1, settings.evidence_mcp_concurrency))

        async def fetch_one(index: int, sha: str) -> None:
            async with limiter:
                details[index] = await self._safe_call_github_tool(
                    session,
                    settings.github_mcp_commit_tool,
                    {
                        "owner": owner,
                        "repo": repo,
                        "sha": sha,
                        "detail": "full_patch",
                    },
                )

        async with anyio.create_task_group() as task_group:
            for index, sha in enumerate(commit_shas):
                task_group.start_soon(fetch_one, index, sha)

        return [detail for detail in details if detail is not None]

    async def _fetch_github_commits(
        self,
        *,
        session: ClientSession,
        owner: str,
        repo: str,
        github_login: str | None,
        github_verified_emails: list[str] | None,
        max_commits: int,
    ) -> tuple[dict, list[str]]:
        """검증된 login·이메일의 커밋을 중복 없이 최대 개수까지 수집한다."""
        per_page = min(100, max(1, max_commits))
        pages: list[dict] = []
        shas: list[str] = []
        authors: list[str] = []
        seen_authors: set[str] = set()
        for author in [*(github_verified_emails or []), github_login]:
            if not isinstance(author, str) or not author.strip():
                continue
            normalized = author.strip()
            key = normalized.casefold()
            if key not in seen_authors:
                seen_authors.add(key)
                authors.append(normalized)

        for author in authors:
            page = 1
            while len(shas) < max_commits:
                requested_count = min(per_page, max_commits - len(shas))
                arguments: dict[str, object] = {
                    "owner": owner,
                    "repo": repo,
                    "author": author,
                    "page": page,
                    "perPage": requested_count,
                }
                response = await self._safe_call_github_tool(
                    session,
                    settings.github_mcp_commits_tool,
                    arguments,
                )
                pages.append(response)
                page_shas = _extract_github_commit_shas(response)
                new_shas = [sha for sha in page_shas if sha not in shas]
                shas.extend(new_shas)

                if (
                    response.get("isError")
                    or not page_shas
                    or len(page_shas) < requested_count
                ):
                    break
                page += 1
            if len(shas) >= max_commits:
                break

        return {"pages": pages}, shas[:max_commits]

    async def _fetch_github_directory_tree(
        self,
        session: ClientSession,
        owner: str,
        repo: str,
        max_dirs: int = 100,
        max_depth: int = 6,
    ) -> dict:
        """get_file_contents directory 응답을 제한적으로 재귀 조회해 tree 텍스트를 만든다."""
        root = await self._safe_call_github_tool(
            session,
            settings.github_mcp_contents_tool,
            {"owner": owner, "repo": repo, "path": "/"},
        )
        directory_responses: dict[str, dict] = {}
        queue = _extract_github_directory_paths(root, parent_path="")
        visited: set[str] = set()

        concurrency = max(1, settings.evidence_mcp_concurrency)
        while queue and len(visited) < max_dirs:
            batch: list[str] = []
            while queue and len(batch) < concurrency and len(visited) + len(batch) < max_dirs:
                path = queue.pop(0)
                if (
                    path in visited
                    or path in batch
                    or path.count("/") >= max_depth
                ):
                    continue
                batch.append(path)

            if not batch:
                continue

            responses: dict[str, dict] = {}

            async def fetch_directory(path: str) -> None:
                responses[path] = await self._safe_call_github_tool(
                    session,
                    settings.github_mcp_contents_tool,
                    {"owner": owner, "repo": repo, "path": path},
                )

            async with anyio.create_task_group() as task_group:
                for path in batch:
                    task_group.start_soon(fetch_directory, path)

            for path in batch:
                visited.add(path)
                response = responses[path]
                directory_responses[path] = response
                for child_path in _extract_github_directory_paths(
                    response,
                    parent_path=path,
                ):
                    if child_path not in visited and child_path not in queue:
                        queue.append(child_path)

        return _combine_github_tree_response(root, directory_responses)

    async def _safe_call_github_tool(
        self,
        session: ClientSession,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """GitHub MCP tool 1건을 호출하고 실패를 isError 응답으로 정규화한다."""
        try:
            result = await session.call_tool(tool_name, arguments=arguments)
        except Exception as exc:
            return {
                "isError": True,
                "content": [{"type": "text", "text": str(exc)}],
            }
        return result.model_dump(mode="json")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """중첩 비동기 예외를 포함해 GitHub MCP 429 응답인지 확인한다."""
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_rate_limit_error(child) for child in exc.exceptions)
    message = str(exc)
    return "429" in message or "Too Many Requests" in message


def _extract_github_commit_shas(commits: dict) -> list[str]:
    """GitHub MCP list_commits 응답에서 최상위 commit sha만 추출한다.

    응답 전체를 재귀 순회하면 parent/tree SHA가 사용자 commit으로 섞일 수 있다.
    따라서 구조화된 commit 레코드의 ``sha`` 필드와 텍스트 응답의 명시적인
    commit/sha 필드만 허용한다.
    """
    if commits.get("isError"):
        return []

    candidates: list[str] = []
    structured = commits.get("structuredContent")
    if structured is not None:
        candidates.extend(_extract_commit_record_shas(structured))

    for item in commits.get("content", []):
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                candidates.extend(_extract_commit_shas_from_text(text))

    shas: list[str] = []
    seen: set[str] = set()
    for sha in candidates:
        if sha not in seen:
            seen.add(sha)
            shas.append(sha)
    return shas


def _extract_commit_record_shas(value: object) -> list[str]:
    """알려진 list 응답 컨테이너에서 commit 레코드의 sha만 읽는다."""
    if isinstance(value, list):
        return [
            sha
            for item in value
            if isinstance(item, dict)
            for sha in [_valid_full_sha(item.get("sha"))]
            if sha is not None
        ]

    if not isinstance(value, dict):
        return []

    direct_sha = _valid_full_sha(value.get("sha"))
    if direct_sha is not None:
        return [direct_sha]

    for key in ("commits", "items", "results", "result", "data"):
        child = value.get(key)
        if isinstance(child, (dict, list)):
            shas = _extract_commit_record_shas(child)
            if shas:
                return shas
    return []


def _extract_commit_shas_from_text(text: str) -> list[str]:
    """JSON 또는 줄 단위 텍스트에서 명시적인 commit SHA만 추출한다."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if payload is not None:
        shas = _extract_commit_record_shas(payload)
        if shas:
            return shas

    patterns = (
        r'(?im)^\s*commit\s+([0-9a-f]{40})\b',
        r'(?im)^\s*sha\s*[:=]\s*["\']?([0-9a-f]{40})\b',
        r'(?im)^\s*["\']sha["\']\s*:\s*["\']([0-9a-f]{40})["\']',
    )
    return [match.group(1) for pattern in patterns for match in re.finditer(pattern, text)]


def _valid_full_sha(value: object) -> str | None:
    """값이 40자리 Git SHA일 때만 정규화해 반환한다."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value, re.IGNORECASE):
        return None
    return value.lower()


def _combine_github_tree_response(root: dict, directory_responses: dict[str, dict]) -> dict:
    """루트와 하위 디렉터리 응답을 하나의 MCP text 응답 형태로 합친다."""
    parts = [_extract_github_response_text(root)]
    for path, response in directory_responses.items():
        text = _extract_github_response_text(response)
        if text:
            parts.append(f"\n# {path}\n{text}")

    return {
        "content": [
            {
                "type": "text",
                "text": "\n".join(part for part in parts if part),
            }
        ],
        "root": root,
        "directories": directory_responses,
    }


def _extract_github_response_text(response: object) -> str:
    """GitHub MCP 응답을 directory/path 추출용 텍스트로 직렬화한다."""
    if not isinstance(response, dict):
        return json.dumps(response, ensure_ascii=False)
    if response.get("isError"):
        return ""

    texts: list[str] = []
    for item in response.get("content", []):
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    if texts:
        return "\n".join(texts)

    structured = response.get("structuredContent")
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)

    return json.dumps(response, ensure_ascii=False)


def _extract_github_directory_paths(response: object, parent_path: str) -> list[str]:
    """GitHub MCP directory 응답에서 하위 directory path 후보를 추출한다."""
    paths: list[str] = []
    seen: set[str] = set()

    if isinstance(response, dict):
        structured = response.get("structuredContent")
        if structured is not None:
            for path in _extract_directory_paths_from_node(structured, parent_path):
                if _is_allowed_github_directory(path) and path not in seen:
                    seen.add(path)
                    paths.append(path)

    text = _extract_github_response_text(response)
    for path in _extract_directory_paths_from_text(text, parent_path):
        if _is_allowed_github_directory(path) and path not in seen:
            seen.add(path)
            paths.append(path)

    return paths


def _extract_directory_paths_from_node(value: object, parent_path: str) -> list[str]:
    """structuredContent의 dict/list에서 directory path를 찾는다."""
    if isinstance(value, list):
        paths: list[str] = []
        for item in value:
            paths.extend(_extract_directory_paths_from_node(item, parent_path))
        return paths

    if not isinstance(value, dict):
        return []

    paths = []
    item_type = str(value.get("type") or "").lower()
    path_value = value.get("path")
    name_value = value.get("name")
    if item_type in {"dir", "directory", "tree"}:
        if isinstance(path_value, str) and path_value:
            paths.append(path_value.strip("/"))
        elif isinstance(name_value, str) and name_value:
            paths.append(_join_github_path(parent_path, name_value))

    for child in value.values():
        paths.extend(_extract_directory_paths_from_node(child, parent_path))
    return paths


def _extract_directory_paths_from_text(text: str, parent_path: str) -> list[str]:
    """MCP text 응답에서 directory로 보이는 path 후보를 찾는다."""
    paths: list[str] = []

    for match in re.finditer(
        r'"path"\s*:\s*"([^"]+)"[^{}]{0,160}"type"\s*:\s*"(?:dir|directory|tree)"',
        text,
        re.IGNORECASE,
    ):
        paths.append(match.group(1).strip("/"))
    for match in re.finditer(
        r'"type"\s*:\s*"(?:dir|directory|tree)"[^{}]{0,160}"path"\s*:\s*"([^"]+)"',
        text,
        re.IGNORECASE,
    ):
        paths.append(match.group(1).strip("/"))

    for line in text.splitlines():
        stripped = line.strip().strip("-* ")
        if not stripped:
            continue
        if re.search(r"\.[A-Za-z0-9]+$", stripped):
            continue
        if re.search(r"\b(dir|directory|folder)\b", stripped, re.IGNORECASE):
            candidate = stripped.split()[-1].strip("/\"'")
            paths.append(_join_github_path(parent_path, candidate))
        elif stripped.endswith("/"):
            paths.append(_join_github_path(parent_path, stripped.strip("/")))

    return paths


def _join_github_path(parent_path: str, child_path: str) -> str:
    """GitHub repository 내부 path를 slash 기준으로 합친다."""
    child_path = child_path.strip("/")
    if not parent_path:
        return child_path
    if child_path.startswith(f"{parent_path}/"):
        return child_path
    return f"{parent_path.strip('/')}/{child_path}"


def _is_allowed_github_directory(path: str) -> bool:
    """재귀 조회 대상 directory인지 판단한다."""
    if not path or path.startswith(("http://", "https://", "github.com")):
        return False
    parts = {part.lower() for part in path.split("/")}
    excluded = {
        ".git",
        ".github",
        ".idea",
        ".vscode",
        "__pycache__",
        "build",
        "dist",
        "generated",
        "node_modules",
        "target",
    }
    return not bool(parts & excluded)
