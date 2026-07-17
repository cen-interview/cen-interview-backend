"""GitHub OAuth 이메일 권한과 검증 이메일 선별을 테스트한다."""

from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from interview.api.auth.service import (
    _fetch_verified_github_emails,
    build_github_oauth_authorize_url,
    exchange_github_oauth_code,
)


def test_github_authorize_url_requests_user_email_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        "interview.api.auth.service.settings.github_oauth_client_id",
        "client-id",
    )
    monkeypatch.setattr(
        "interview.api.auth.service.settings.github_oauth_scope",
        "read:user user:email repo",
    )
    user = SimpleNamespace(id=1, email="user@example.com")

    result = build_github_oauth_authorize_url(user=user)
    query = parse_qs(urlparse(result["authorize_url"]).query)

    assert query["scope"] == ["read:user user:email repo"]


def test_fetch_verified_github_emails_keeps_only_verified_and_deduplicates(
    monkeypatch,
) -> None:
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> list[dict]:
            return [
                {"email": "primary@example.com", "verified": True, "primary": True},
                {"email": "PRIMARY@example.com", "verified": True, "primary": False},
                {"email": "unverified@example.com", "verified": False},
                {"email": "second@example.com", "verified": True},
            ]

    monkeypatch.setattr(
        "interview.api.auth.service.httpx.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    assert _fetch_verified_github_emails("token") == [
        "primary@example.com",
        "second@example.com",
    ]


def test_exchange_github_oauth_code_persists_verified_emails(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, payload, status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

    class FakeQuery:
        def filter(self, *args):
            _ = args
            return self

        @staticmethod
        def first():
            return None

    class FakeDb:
        def __init__(self) -> None:
            self.added = None

        @staticmethod
        def query(*args):
            _ = args
            return FakeQuery()

        def add(self, value) -> None:
            self.added = value

        @staticmethod
        def commit() -> None:
            return None

        @staticmethod
        def refresh(value) -> None:
            value.id = 10

    monkeypatch.setattr(
        "interview.api.auth.service.settings.github_oauth_client_id",
        "client-id",
    )
    monkeypatch.setattr(
        "interview.api.auth.service.settings.github_oauth_client_secret",
        "client-secret",
    )
    monkeypatch.setattr(
        "interview.api.auth.service.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            {
                "access_token": "access-token",
                "token_type": "bearer",
                "scope": "read:user,user:email,repo",
            }
        ),
    )

    def fake_get(url: str, **kwargs):
        _ = kwargs
        if url.endswith("/user/emails"):
            return FakeResponse(
                [{"email": "commit@example.com", "verified": True}]
            )
        return FakeResponse({"id": 123, "login": "octocat"})

    monkeypatch.setattr("interview.api.auth.service.httpx.get", fake_get)
    db = FakeDb()

    result = exchange_github_oauth_code(db=db, user_id=1, code="oauth-code")

    assert db.added.verified_emails == ["commit@example.com"]
    assert result["verified_emails"] == ["commit@example.com"]
