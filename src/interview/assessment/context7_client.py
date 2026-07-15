"""Context7 documentation client used by final report generation.

The client uses Context7's HTTP API directly so the backend can fetch
documentation when an interview report is generated. OpenAI credentials and
Context7 credentials are intentionally kept separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from interview.config import settings


CONTEXT7_API_BASE_URL = "https://context7.com/api/v2"
DEFAULT_TIMEOUT_SECONDS = 10.0


class Context7Error(RuntimeError):
    """Raised when Context7 cannot return documentation."""


@dataclass(frozen=True)
class Context7Client:
    """Small synchronous client for Context7 library search and context APIs."""

    api_key: str = ""
    base_url: str = CONTEXT7_API_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.api_key:
            configured_key = getattr(settings, "context7_api_key", "")
            object.__setattr__(self, "api_key", configured_key)

    def search_library(
        self,
        library_name: str,
        query: str,
    ) -> list[dict]:
        """Find Context7 library IDs matching a topic or package name."""
        payload = self._get_json(
            "/libs/search",
            {
                "libraryName": library_name,
                "query": query,
            },
        )

        results = payload.get("results", [])
        return results if isinstance(results, list) else []

    def get_context(
        self,
        library_id: str,
        query: str,
    ) -> dict:
        """Fetch current documentation snippets for a Context7 library ID."""
        return self._get_json(
            "/context",
            {
                "libraryId": library_id,
                "query": query,
                "type": "json",
            },
        )

    def fetch_topic_context(
        self,
        topic: str,
        query: str,
    ) -> dict | None:
        """Resolve a topic to a library and fetch its latest documentation."""
        libraries = self.search_library(
            library_name=topic,
            query=query,
        )

        if not libraries:
            return None

        library_id = libraries[0].get("id")
        if not library_id:
            return None

        context = self.get_context(
            library_id=library_id,
            query=query,
        )

        return {
            "library_id": library_id,
            "library": libraries[0],
            "context": context,
        }

    def _get_json(
        self,
        path: str,
        params: dict[str, str],
    ) -> dict:
        url = f"{self.base_url}{path}?{urlencode(params)}"
        headers = {"Accept": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            url=url,
            headers=headers,
            method="GET",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise Context7Error(
                f"Context7 request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise Context7Error(
                f"Context7 request could not be completed: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise Context7Error("Context7 returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise Context7Error("Context7 returned an unexpected response shape.")

        return payload


def get_context7_client() -> Context7Client:
    """Return a client configured from application settings."""
    return Context7Client()
