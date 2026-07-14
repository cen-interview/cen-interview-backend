"""외부 소스 접근 (Notion MCP / GitHub MCP).

인덱싱 파이프라인이 원본을 긁어오는 부분만 담당한다. 추출/청킹/저장은 각각
extract / chunking / store 가 맡는다 (한 파일에 다 넣지 않는다).

⚠️ 면접이 시작되면 이 모듈은 동작하지 않는다. 면접 전 1회만 호출.
"""
import base64
import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

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
    payload = _extract_notion_payload(response)
    raw_text = _extract_notion_text(response, payload).strip()
    if not raw_text:
        return []

    title = payload.get("title") or _extract_notion_title(raw_text, root_link)
    week = _extract_week(title, raw_text)
    date = _extract_date(raw_text)
    doc_type = _infer_doc_type(title, raw_text)
    source_url = payload.get("url") or root_link
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    return [
        RawDoc(
            source_url=source_url,
            source_type="notion",
            title=title,
            raw_text=raw_text,
            meta={
                "root_link": root_link,
                "tool": "notion-fetch",
                "notion_type": metadata.get("type"),
                "doc_type": doc_type,
                "week": week,
                "date": date,
            },
        )
    ]


def _normalize_notion_link(link: str) -> str:
    """중복 방문 판단을 위해 Notion 링크의 불필요한 끝 문자를 정리한다."""
    normalized = link.strip()
    if normalized.startswith("{{") and normalized.endswith("}}"):
        normalized = normalized[2:-2].strip()
    normalized = normalized.rstrip("/")

    parsed = urlsplit(normalized)
    if parsed.netloc == "app.notion.com" and parsed.path.startswith("/p/"):
        page_slug = parsed.path.removeprefix("/p/")
        match = re.search(r"([0-9a-f]{32})$", page_slug, re.IGNORECASE)
        if match:
            normalized_path = f"/p/{match.group(1)}"
            return urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    normalized_path,
                    parsed.query,
                    parsed.fragment,
                )
            )

    return normalized


def _extract_notion_payload(response: dict) -> dict:
    """MCP text content 안에 JSON wrapper가 있으면 dict로 파싱한다."""
    for item in response.get("content", []):
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        if not isinstance(text, str):
            continue

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict):
            return payload

    return {}


def _extract_child_page_links(raw_text: str, exclude_links: set[str] | None = None) -> list[str]:
    """Notion enhanced Markdown 안의 child page URL 목록을 추출한다."""
    links: list[str] = []
    seen: set[str] = set(exclude_links or set())

    patterns = [
        r'<page\s+url="([^"]+)"',
        r'https://(?:app\.)?notion\.com/p/_[^\s"<>)}]+',
        r'https://(?:www\.)?notion\.so/[^\s"<>)}]+',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, raw_text):
            link = match.group(1) if match.groups() else match.group(0)
            link = _normalize_notion_link(link)
            if link and link not in seen:
                seen.add(link)
                links.append(link)

    return links


def _extract_database_row_page_ids(raw_text: str) -> list[str]:
    """Query 결과 안의 Notion page UUID 후보를 추출한다."""
    page_ids: list[str] = []
    seen: set[str] = set()

    for match in re.finditer(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        raw_text,
        re.IGNORECASE,
    ):
        link = _normalize_notion_link(match.group(0))
        if link.startswith("collection://"):
            continue
        if link and link not in seen:
            seen.add(link)
            page_ids.append(link)

    return page_ids


def _extract_data_source_urls(raw_text: str) -> list[str]:
    """Notion database 응답 안의 data source URL 목록을 추출한다."""
    urls: list[str] = []
    seen: set[str] = set()

    for match in re.finditer(r'<data-source\s+url="([^"]+)"', raw_text):
        url = _normalize_notion_link(match.group(1))
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def _is_database_view_url(link: str) -> bool:
    """Notion URL이 database view 식별자(v=...)를 포함하는지 확인한다."""
    return "?v=" in link or "&v=" in link


def _extract_notion_text(response: dict, payload: dict | None = None) -> str:
    """MCP call_tool 응답에서 Notion page/database 본문만 최대한 안전하게 꺼낸다."""
    if payload:
        text = payload.get("text")
        if isinstance(text, str):
            return text

    texts: list[str] = []

    for item in response.get("content", []):
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        if isinstance(text, str):
            texts.append(text)

    if texts:
        return "\n\n".join(texts)

    structured = response.get("structuredContent")
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, indent=2)

    return json.dumps(response, ensure_ascii=False, indent=2)


def _extract_notion_title(raw_text: str, root_link: str) -> str:
    """Markdown heading 또는 page tag에서 제목을 추정한다."""
    for line in raw_text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()

    match = re.search(r'title="([^"]+)"', raw_text)
    if match:
        return match.group(1).strip()

    return root_link.rstrip("/").split("/")[-1] or "Notion page"


def _extract_week(title: str, raw_text: str) -> int | None:
    match = re.search(r"(\d+)\s*주차", f"{title}\n{raw_text}")
    if match:
        return int(match.group(1))
    return None


def _extract_date(raw_text: str) -> str | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}", raw_text)
    if match:
        return match.group(0)
    return None


def _infer_doc_type(title: str, raw_text: str) -> str:
    text = f"{title}\n{raw_text}".lower()

    if "회고" in text or "retrospective" in text:
        return "retrospective"
    if "주차" in text or "학습" in text:
        return "weekly_note"
    if "트러블슈팅" in text or "troubleshooting" in text:
        return "troubleshooting"
    if "개념" in text:
        return "concept"
    return "notion_page"


def _github_response_to_raw_docs(response: dict, repo_link: str) -> list[RawDoc]:
    """GitHub MCP 원본 응답을 Evidence 파이프라인의 RawDoc 목록으로 변환한다.

    Args:
        response: EvidenceMcpClient가 GitHub MCP tool call로 모은 원본 응답.
            repository/search 결과, readme, tree 키를 가진 dict를 기대한다.
        repo_link: canonical GitHub 저장소 링크.

    Returns:
        README, 저장소 메타, 디렉터리 트리를 담은 GitHub RawDoc 목록.
    """
    owner = response.get("owner")
    repo = response.get("repo")
    repo_name = f"{owner}/{repo}" if owner and repo else repo_link.rstrip("/").split("/")[-1]
    docs: list[RawDoc] = []

    readme_text = _extract_mcp_text(response.get("readme"))
    if readme_text and not _is_github_download_status_text(readme_text):
        docs.append(
            RawDoc(
                source_url=f"{repo_link}#readme",
                source_type="github",
                title=f"{repo_name} README",
                raw_text=readme_text,
                meta={
                    "repo": repo_name,
                    "doc_type": "README",
                    "file_path": "README.md",
                },
            )
        )

    repository_text = _extract_mcp_text(response.get("repository"))
    if repository_text:
        docs.append(
            RawDoc(
                source_url=repo_link,
                source_type="github",
                title=f"{repo_name} repository metadata",
                raw_text=repository_text,
                meta={
                    "repo": repo_name,
                    "doc_type": "repository_meta",
                },
            )
        )

    tree_text = _extract_mcp_text(response.get("tree"))
    if tree_text:
        docs.append(
            RawDoc(
                source_url=f"{repo_link}/tree/HEAD",
                source_type="github",
                title=f"{repo_name} directory tree",
                raw_text=tree_text,
                meta={
                    "repo": repo_name,
                    "doc_type": "directory_tree",
                },
            )
        )

    return docs


def _github_code_response_to_raw_docs(
    *,
    repo_link: str,
    owner: str,
    repo: str,
    file_contents: dict[str, dict],
    touched_files: dict[str, list[str]],
    github_login: str | None,
) -> list[RawDoc]:
    """선별된 GitHub 코드 파일 응답을 code RawDoc 목록으로 변환한다."""
    repo_name = f"{owner}/{repo}"
    docs: list[RawDoc] = []

    for file_path, response in file_contents.items():
        raw_text = _extract_github_file_content_text(response)
        if not raw_text:
            continue

        commit_shas = touched_files.get(file_path, [])
        docs.append(
            RawDoc(
                source_url=f"{repo_link}/blob/HEAD/{file_path}",
                source_type="github",
                title=f"{repo_name} {file_path}",
                raw_text=raw_text,
                meta={
                    "repo": repo_name,
                    "doc_type": "code",
                    "file_path": file_path,
                    "language": _infer_github_language(file_path),
                    "ownership": "user_touched" if commit_shas else "repo_context",
                    "author_login": github_login,
                    "commit_shas": commit_shas,
                    "commit_count": len(commit_shas),
                },
            )
        )

    return docs


def _extract_github_file_content_text(response: object) -> str:
    """GitHub MCP file contents 응답에서 실제 파일 본문을 우선 추출한다."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if not isinstance(response, dict) or response.get("isError"):
        return ""

    resource_text = _extract_mcp_resource_text(response)
    if resource_text:
        return resource_text

    structured_text = _extract_structured_file_text(response.get("structuredContent"))
    if structured_text:
        return structured_text

    text = _extract_mcp_text(response)
    if _is_github_download_status_text(text):
        return ""
    return text


def _extract_mcp_resource_text(response: dict) -> str:
    """MCP embedded resource content에서 text/blob 값을 추출한다."""
    for item in response.get("content", []):
        if not isinstance(item, dict):
            continue
        resource = item.get("resource")
        if not isinstance(resource, dict):
            continue

        text = resource.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        blob = resource.get("blob")
        if isinstance(blob, str) and blob.strip():
            try:
                return base64.b64decode(blob).decode("utf-8").strip()
            except (ValueError, UnicodeDecodeError):
                return blob.strip()

    return ""


def _extract_structured_file_text(value: object) -> str:
    """structuredContent에서 파일 본문으로 쓰일 만한 문자열을 추출한다."""
    if not isinstance(value, dict):
        return ""

    for key in ("content", "text", "fileContent", "file_content", "body"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            if key == "content" and value.get("encoding") == "base64":
                try:
                    return base64.b64decode(text).decode("utf-8").strip()
                except (ValueError, UnicodeDecodeError):
                    return text.strip()
            return text.strip()

    for child in value.values():
        nested = _extract_structured_file_text(child)
        if nested:
            return nested

    return ""


def _is_github_download_status_text(text: str) -> bool:
    """GitHub MCP의 파일 다운로드 상태 메시지인지 판단한다."""
    normalized = text.strip().lower()
    return normalized.startswith("successfully downloaded text file")


def _extract_mcp_text(value: object) -> str:
    """MCP tool 응답에서 RawDoc에 넣을 텍스트를 추출한다."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    if value.get("isError"):
        return ""

    texts: list[str] = []
    for item in value.get("content", []):
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    if texts:
        return "\n\n".join(texts)

    structured = value.get("structuredContent")
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, indent=2)

    return json.dumps(value, ensure_ascii=False, indent=2)


def _extract_github_file_paths(tree_response: object) -> list[str]:
    """GitHub MCP directory 응답에서 파일 경로 후보를 추출한다."""
    tree_text = _extract_mcp_text(tree_response)
    if not tree_text:
        return []

    paths: list[str] = []
    seen: set[str] = set()
    current_dir = ""

    for line in tree_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            current_dir = stripped[2:].strip().strip("/")
            continue

        for match in re.finditer(r"[\w./@+\-=]+(?:\.[A-Za-z0-9]+)", stripped):
            path = match.group(0).strip("./")
            if current_dir and "/" not in path:
                path = f"{current_dir}/{path}"
            if _looks_like_github_file_path(path) and path not in seen:
                seen.add(path)
                paths.append(path)

    if paths:
        return paths

    for match in re.finditer(r"[\w./@+\-=]+(?:\.[A-Za-z0-9]+)", tree_text):
        path = match.group(0).strip("./")
        if _looks_like_github_file_path(path) and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _looks_like_github_file_path(path: str) -> bool:
    """디렉터리 출력에서 실제 repository 파일 경로처럼 보이는지 판단한다."""
    if "/" not in path and "." not in path:
        return False
    if path.startswith(("http", "github.com")):
        return False
    return bool(re.search(r"\.[A-Za-z0-9]+$", path))


SOURCE_EXTENSIONS = {
    ".java": "java",
    ".kt": "kotlin",
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
}

EXCLUDED_PATH_PARTS = {
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "__pycache__",
    "__tests__",
    "build",
    "dist",
    "generated",
    "node_modules",
    "target",
    "test",
    "tests",
}

EXCLUDED_FILENAMES = {
    "data.sql",
    "dummy-users.sql",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "uv.lock",
    "poetry.lock",
    "gradlew",
    "gradlew.bat",
}


def _is_github_source_file(path: str) -> bool:
    """임베딩 후보가 될 소스 파일인지 판단한다."""
    return _infer_github_language(path) is not None and not _is_github_excluded_path(path)


def _is_github_excluded_path(path: str) -> bool:
    """테스트/빌드/락/설정 위주의 파일을 제외한다."""
    parts = {part.lower() for part in path.split("/")}
    if parts & EXCLUDED_PATH_PARTS:
        return True
    filename = path.rsplit("/", 1)[-1].lower()
    if filename in EXCLUDED_FILENAMES:
        return True
    if filename.startswith(("dummy", "seed")):
        return True
    if filename.endswith((".lock", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico")):
        return True
    return False


def _infer_github_language(path: str) -> str | None:
    """파일 확장자로 GitHub 코드 언어를 추정한다."""
    lower_path = path.lower()
    for extension, language in SOURCE_EXTENSIONS.items():
        if lower_path.endswith(extension):
            return language
    return None


def _build_github_touched_file_map(commit_details: object) -> dict[str, list[str]]:
    """get_commit detail 응답에서 파일별 commit sha 목록을 만든다."""
    touched: dict[str, list[str]] = {}
    for detail in commit_details if isinstance(commit_details, list) else []:
        if not isinstance(detail, dict) or detail.get("isError"):
            continue
        text = _extract_mcp_text(detail)
        sha = _extract_first_sha(text)
        for file_path in _extract_github_file_paths_from_commit_text(text):
            if not _is_github_source_file(file_path):
                continue
            touched.setdefault(file_path, [])
            if sha and sha not in touched[file_path]:
                touched[file_path].append(sha)
    return touched


def _extract_first_sha(text: str) -> str | None:
    match = re.search(r"\b[0-9a-f]{40}\b", text, re.IGNORECASE)
    return match.group(0) if match else None


def _extract_github_file_paths_from_commit_text(text: str) -> list[str]:
    """commit detail 텍스트에서 변경 파일 경로를 추출한다."""
    paths: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"[\w./@+\-=]+(?:\.[A-Za-z0-9]+)", text):
        path = match.group(0).strip("./")
        if _is_github_source_file(path) and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _select_github_code_paths(
    tree_paths: list[str],
    touched_files: dict[str, list[str]],
    max_files: int = 15,
) -> list[str]:
    """전체 핵심 소스와 사용자 touched 파일을 함께 고려해 코드 파일을 선별한다."""
    candidates = [path for path in tree_paths if _is_github_source_file(path)]
    touched_candidates = [path for path in touched_files if path in candidates]

    selected: list[str] = []
    for path in sorted(touched_candidates, key=_github_path_priority):
        if path not in selected:
            selected.append(path)

    for path in sorted(candidates, key=_github_path_priority):
        if len(selected) >= max_files:
            break
        if path not in selected:
            selected.append(path)

    return selected[:max_files]


def _github_path_priority(path: str) -> tuple[int, int, int, int, int, str]:
    """핵심 구현 파일이 먼저 오도록 정렬 우선순위를 계산한다."""
    lower = path.lower()
    language = _infer_github_language(path)
    important_names = (
        "controller",
        "service",
        "repository",
        "config",
        "security",
        "auth",
        "model",
        "entity",
        "schema",
        "router",
        "component",
        "page",
    )
    main_source_score = 0 if re.search(r"/src/main/(java|kotlin|python|ts|tsx|js|jsx)/", f"/{lower}") else 1
    language_score = 0 if language not in {"html", "css", "scss", "sql"} else 1
    resource_score = 1 if "/resources/" in lower else 0
    name_score = 0 if any(name in lower for name in important_names) else 1
    depth = path.count("/")
    return (main_source_score, language_score, resource_score, name_score, depth, path)


def _parse_github_repo_url(repo_link: str) -> tuple[str, str]:
    """GitHub repository 링크에서 owner/repo를 추출한다.

    지원하는 입력:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - github.com/owner/repo
      - git@github.com:owner/repo.git
    """
    link = repo_link.strip()
    if not link:
        raise ValueError("GitHub repository URL이 비어 있습니다.")

    ssh_match = re.match(r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", link)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)

    if link.startswith("github.com/"):
        link = f"https://{link}"

    parsed = urlsplit(link)
    if parsed.netloc.lower() != "github.com":
        raise ValueError("GitHub repository URL이 아닙니다.")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("owner/repo 형식의 GitHub URL이 아닙니다.")

    owner = parts[0]
    repo = parts[1].removesuffix(".git")
    if not owner or not repo:
        raise ValueError("owner/repo 형식의 GitHub URL이 아닙니다.")

    return owner, repo


def _canonical_github_repo_url(owner: str, repo: str) -> str:
    """MCP 호출과 RawDoc source_url에 사용할 표준 GitHub repository URL을 만든다."""
    return f"https://github.com/{owner}/{repo}"


class NotionSource:
    """Notion MCP 로 사용자가 등록한 학습 기록 링크 목록을 가져온다."""

    def __init__(
        self,
        mcp_client: EvidenceMcpClient | None = None,
        max_pages: int = 100,
        max_depth: int = 3,
    ) -> None:
        """MCP client와 Notion 재귀 수집 제한값을 초기화한다."""
        self.mcp_client = mcp_client or EvidenceMcpClient()
        self.max_pages = max_pages
        self.max_depth = max_depth

    def fetch_pages(self, root_links: list[str]) -> list[RawDoc]:
        """등록된 Notion 링크를 순회하며 RawDoc 리스트로 반환한다.

        사용자가 등록한 실제 Notion page 링크를 notion-fetch로 수집한다.
        응답에 child page URL이 포함된 경우에는 제한적으로 재귀 fetch한다.
        database/timeline row 자동 조회는 MCP 요금제 제한이 있을 수 있으므로
        best-effort로만 시도한다.
        """
        visited: set[str] = set()
        raw_docs: list[RawDoc] = []
        for root_link in root_links:
            raw_docs.extend(self._fetch_notion_entity(root_link, visited, depth=0))
        return raw_docs

    def _fetch_notion_entity(
        self,
        link: str,
        visited: set[str],
        depth: int,
    ) -> list[RawDoc]:
        """Notion page를 fetch하고 응답에 포함된 child page를 제한적으로 재귀 수집한다."""
        normalized_link = _normalize_notion_link(link)
        if normalized_link in visited:
            return []
        if len(visited) >= self.max_pages:
            return []
        if depth > self.max_depth:
            return []

        visited.add(normalized_link)

        try:
            response = self.mcp_client.call_notion_tool(
                root_link=normalized_link,
                tool_name="notion-fetch",
                arguments={"id": normalized_link},
            )
        except Exception:
            return []

        if response.get("isError"):
            return []

        raw_docs = _notion_response_to_raw_docs(response, normalized_link)
        excluded_links = {normalized_link}
        excluded_links.update(_normalize_notion_link(doc.source_url) for doc in raw_docs)

        for raw_doc in list(raw_docs):
            for child_link in _extract_child_page_links(raw_doc.raw_text, excluded_links):
                if len(visited) >= self.max_pages:
                    break
                raw_docs.extend(
                    self._fetch_notion_entity(
                        child_link,
                        visited,
                        depth=depth + 1,
                    )
                )

            for data_source_url in _extract_data_source_urls(raw_doc.raw_text):
                if len(visited) >= self.max_pages:
                    break
                raw_docs.extend(
                    self._fetch_data_source_pages(
                        data_source_url,
                        visited,
                        depth=depth + 1,
                    )
                )

            if _is_database_view_url(normalized_link):
                raw_docs.extend(
                    self._fetch_database_view_pages(
                        normalized_link,
                        visited,
                        depth=depth + 1,
                    )
                )

        return raw_docs

    def _fetch_database_view_pages(
        self,
        view_url: str,
        visited: set[str],
        depth: int,
    ) -> list[RawDoc]:
        """Notion database view에서 row page URL을 찾아 best-effort로 fetch한다.

        Notion MCP의 database/view query tool은 워크스페이스 플랜과 Notion AI
        권한에 따라 실패할 수 있다. 실패해도 등록된 page fetch 흐름은 유지한다.
        """
        if depth > self.max_depth:
            return []

        try:
            response = self.mcp_client.call_notion_tool(
                root_link=view_url,
                tool_name="notion-query-database-view",
                arguments={
                    "view_url": view_url,
                    "page_size": min(self.max_pages, 100),
                    "is_archived": False,
                },
            )
        except Exception:
            return []

        if response.get("isError"):
            return []

        query_text = _extract_notion_text(response, _extract_notion_payload(response))

        raw_docs: list[RawDoc] = []
        for child_link in _extract_child_page_links(query_text):
            if len(visited) >= self.max_pages:
                break
            raw_docs.extend(
                self._fetch_notion_entity(
                    child_link,
                    visited,
                    depth=depth + 1,
                )
            )

        return raw_docs

    def _fetch_data_source_pages(
        self,
        data_source_url: str,
        visited: set[str],
        depth: int,
    ) -> list[RawDoc]:
        """Notion data source에서 row page URL을 찾아 best-effort로 fetch한다.

        notion-query-data-sources는 Business plan 이상 및 Notion AI 권한을
        요구할 수 있으므로 page fetch 흐름과 분리된 보조 경로로 다룬다.
        """
        if depth > self.max_depth:
            return []

        try:
            response = self.mcp_client.call_notion_tool(
                root_link=data_source_url,
                tool_name="notion-query-data-sources",
                arguments={
                    "data": {
                        "data_source_urls": [data_source_url],
                        "query": f'SELECT * FROM "{data_source_url}" LIMIT {self.max_pages}',
                    }
                },
            )
        except Exception:
            return []

        if response.get("isError"):
            return []

        query_text = _extract_notion_text(response, _extract_notion_payload(response))

        raw_docs: list[RawDoc] = []
        for child_link in _extract_child_page_links(query_text):
            if len(visited) >= self.max_pages:
                break
            raw_docs.extend(
                self._fetch_notion_entity(
                    child_link,
                    visited,
                    depth=depth + 1,
                )
            )

        return raw_docs


class GitHubSource:
    """GitHub MCP 로 사용자가 등록한 프로젝트 저장소 목록을 가져온다."""

    def __init__(self, mcp_client: EvidenceMcpClient | None = None) -> None:
        """MCP client를 주입받아 테스트와 실제 tool call 경계를 분리한다."""
        self.mcp_client = mcp_client or EvidenceMcpClient()

    def fetch_repos(
        self,
        repo_links: list[str],
        github_login: str | None = None,
    ) -> list[RawDoc]:
        """등록된 GitHub repository 링크를 정규화하고 RawDoc 리스트로 반환한다.

        등록된 링크를 제한 없이 순회하고 owner/repo를 파싱한다. 잘못된
        GitHub 링크는 전체 수집을 중단하지 않고 스킵한다. tree와 commit
        정보를 바탕으로 핵심 코드 파일도 RawDoc으로 만든다.
        """
        raw_docs: list[RawDoc] = []
        for repo_link in repo_links:
            try:
                owner, repo = _parse_github_repo_url(repo_link)
            except ValueError:
                continue

            canonical_url = _canonical_github_repo_url(owner, repo)
            response = self.mcp_client.call_github_tool(
                canonical_url,
                owner=owner,
                repo=repo,
                github_login=github_login,
            )
            raw_docs.extend(_github_response_to_raw_docs(response, canonical_url))

            touched_files = _build_github_touched_file_map(response.get("commit_details"))
            tree_paths = _extract_github_file_paths(response.get("tree"))
            selected_paths = _select_github_code_paths(tree_paths, touched_files)
            if not selected_paths:
                continue

            file_contents = self.mcp_client.fetch_github_file_contents(
                owner,
                repo,
                selected_paths,
            )
            raw_docs.extend(
                _github_code_response_to_raw_docs(
                    repo_link=canonical_url,
                    owner=owner,
                    repo=repo,
                    file_contents=file_contents,
                    touched_files=touched_files,
                    github_login=github_login,
                )
            )
        return raw_docs
