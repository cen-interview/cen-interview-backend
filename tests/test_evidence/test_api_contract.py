"""프론트엔드가 소비하는 Evidence API 입력/응답 계약 단위 테스트."""

import pytest
from fastapi import HTTPException

from interview.api.evidence.router import _normalize_source_url
from interview.api.evidence.schema import EvidenceIndexRequest, EvidenceSourceCreateRequest


def test_source_request_accepts_github_url() -> None:
    """마이페이지의 GitHub 저장소 링크 입력을 허용한다."""

    request = EvidenceSourceCreateRequest(
        source_type="github",
        url="https://github.com/MINITCEN/MiniPrj-Bugbug",
    )

    assert _normalize_source_url(request.source_type, str(request.url)) == (
        "https://github.com/MINITCEN/MiniPrj-Bugbug"
    )


def test_source_url_rejects_wrong_provider() -> None:
    """GitHub 등록 API가 다른 도메인 링크를 받지 않는다."""

    with pytest.raises(HTTPException) as exc_info:
        _normalize_source_url("github", "https://notion.so/example")

    assert exc_info.value.status_code == 422


def test_index_request_supports_saved_source_ids() -> None:
    """프론트가 저장된 링크 ID 목록만으로 인덱싱을 요청할 수 있다."""

    request = EvidenceIndexRequest(source_ids=[3, 7])

    assert request.notion_links == []
    assert request.github_links == []
    assert request.source_ids == [3, 7]
