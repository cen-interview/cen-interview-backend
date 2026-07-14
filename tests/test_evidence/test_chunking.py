"""EvidenceChunk 분할 규칙을 검증한다."""

import pytest

from interview.evidence.chunking import chunk
from interview.schemas.evidence import EvidenceChunk, SourceType


def _chunk(
    text: str,
    *,
    chunk_id: str = "parent",
    doc_type: str | None = "study",
) -> EvidenceChunk:
    """테스트용 EvidenceChunk를 만든다."""
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        source_type=SourceType.NOTION,
        source_url="https://example.com/source",
        topic="spring security",
        doc_type=doc_type,
        week=3,
        date="2026-07-13",
        confidence=0.8,
    )


def test_chunk_keeps_short_chunk_unchanged() -> None:
    """max_chars 이하 청크는 ID와 메타데이터를 유지한다."""
    source = _chunk("짧은 근거 본문입니다.")

    result = chunk([source], max_chars=75)

    assert result == [source]


def test_chunk_splits_long_text_on_paragraph_boundaries() -> None:
    """긴 일반 문서는 문단 경계를 우선해 분할한다."""
    text = "\n\n".join(
        [
            "첫 번째 문단은 JWT 인증 흐름과 Access Token 검증 과정을 설명한다.",
            "두 번째 문단은 Refresh Token 재발급 API와 예외 처리 흐름을 설명한다.",
            "세 번째 문단은 Spring Security Filter Chain에 필터를 추가한 이유를 설명한다.",
        ]
    )
    source = _chunk(text, chunk_id="note")

    result = chunk([source], max_chars=70)

    assert [item.chunk_id for item in result] == ["note-1", "note-2", "note-3"]
    assert all(len(item.text) <= 70 for item in result)
    assert result[0].text.startswith("첫 번째 문단")
    assert result[1].text.startswith("두 번째 문단")
    assert result[2].topic == source.topic
    assert result[2].confidence == source.confidence
    assert result[2].source_url == source.source_url


def test_chunk_splits_oversized_paragraph_on_sentence_boundaries() -> None:
    """문단 하나가 너무 길면 문장 경계를 기준으로 나눈다."""
    text = (
        "JWT 인증 구현에서 Access Token 검증을 담당하는 필터를 추가했다. "
        "Refresh Token 재발급 API는 인증 예외 경로에서 분리했다. "
        "만료 토큰은 별도 예외로 처리해 클라이언트가 재발급을 요청하게 했다."
    )
    source = _chunk(text, chunk_id="sentence")

    result = chunk([source], max_chars=65)

    assert [item.chunk_id for item in result] == ["sentence-1", "sentence-2", "sentence-3"]
    assert all(item.text.endswith(".") or item.text.endswith("다.") for item in result)


def test_chunk_splits_code_on_blank_line_boundaries() -> None:
    """코드 문서는 빈 줄 경계를 우선해 분할한다."""
    text = """
public class UserService {
    public void login() {
        issueToken();
    }
}

public class AuthController {
    public void refresh() {
        refreshToken();
    }
}
""".strip()
    source = _chunk(text, chunk_id="code", doc_type="code")

    result = chunk([source], max_chars=95)

    assert [item.chunk_id for item in result] == ["code-1", "code-2"]
    assert "UserService" in result[0].text
    assert "AuthController" in result[1].text
    assert result[0].doc_type == "code"


def test_chunk_preserves_fenced_code_block_when_possible() -> None:
    """max_chars 안에 들어가는 fenced code block은 하나의 단위로 유지한다."""
    text = """
설명 문단입니다.

```java
public class JwtFilter {
    public void doFilter() {}
}
```

후속 설명입니다.
""".strip()
    source = _chunk(text, chunk_id="fenced")

    result = chunk([source], max_chars=75)

    assert len(result) == 3
    assert result[1].text.startswith("```java")
    assert result[1].text.endswith("```")


def test_chunk_rejects_non_positive_max_chars() -> None:
    """max_chars는 양수여야 한다."""
    with pytest.raises(ValueError):
        chunk([_chunk("text")], max_chars=0)
