"""RawDoc에서 면접용 EvidenceChunk 후보를 추출하는 규칙을 검증한다."""

from interview.evidence.extract import extract_evidence
from interview.evidence.sources import RawDoc
from interview.schemas.evidence import EvidenceExtractionDecision, EvidenceSectionDecision


def _raw_doc(
    raw_text: str,
    *,
    source_type: str = "notion",
    title: str = "Spring Security JWT 학습",
    meta: dict | None = None,
) -> RawDoc:
    """테스트용 RawDoc을 만든다."""
    return RawDoc(
        source_url="https://example.com/doc",
        source_type=source_type,
        title=title,
        raw_text=raw_text,
        meta=meta or {},
    )


def test_extract_evidence_skips_empty_or_too_short_text() -> None:
    """빈 문서나 너무 짧은 문서는 근거 후보로 만들지 않는다."""
    assert extract_evidence(_raw_doc("")) == []
    assert extract_evidence(_raw_doc("목차\n- [개요](#개요)\n완료")) == []


def test_extract_evidence_removes_boilerplate_and_toc_lines() -> None:
    """MCP boilerplate와 목차 링크를 제거하고 실제 본문만 남긴다."""
    doc = _raw_doc(
        """
Here is the result of "view" for the Page with URL https://example.com as of 2026-07-13:
목차
- [JWT 인증 절차](#jwt-인증-절차)

JWT 인증 절차는 Access Token 검증과 Refresh Token 재발급 흐름으로 구성된다.
Spring Security Filter Chain에 JWT 필터를 추가해 요청마다 토큰을 검증했다.
구현 과정에서 만료 토큰 처리와 재발급 API를 분리했다.
""",
        meta={"doc_type": "weekly_note", "week": 4, "date": "2026-07-13"},
    )

    chunks = extract_evidence(doc)

    assert len(chunks) == 1
    assert "Here is the result" not in chunks[0].text
    assert "[JWT 인증 절차]" not in chunks[0].text
    assert "JWT 인증 절차" in chunks[0].text


def test_extract_evidence_removes_image_references() -> None:
    """이미지 URL과 실행 결과 이미지는 면접 근거 본문에서 제거한다."""
    doc = _raw_doc(
        """
OAuth 콜백 처리 과정에서 state 검증을 추가했다.
토큰 교환 실패 시 400 응답을 반환하도록 예외 처리를 구현했다.

- 실행 결과
![](https://prod-files-secure.s3.us-west-2.amazonaws.com/example/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256)

이후 MCP credential 저장 여부를 status API로 확인했다.
""",
        meta={"doc_type": "retrospective"},
    )

    chunks = extract_evidence(doc)

    assert len(chunks) == 1
    assert "OAuth 콜백" in chunks[0].text
    assert "prod-files-secure" not in chunks[0].text
    assert "X-Amz-" not in chunks[0].text
    assert "실행 결과" not in chunks[0].text


def test_extract_evidence_skips_directory_tree_raw_doc() -> None:
    """GitHub directory tree JSON 조각은 임베딩 대상 evidence로 만들지 않는다."""
    doc = _raw_doc(
        '{"sha":"abc","url":"https://api.github.com/repos/example/project/contents/src"}',
        source_type="github",
        title="example/project directory tree",
        meta={"doc_type": "directory_tree"},
    )

    assert extract_evidence(doc) == []


def test_extract_evidence_normalizes_topic_from_meta() -> None:
    """meta.topic이 있으면 검색 필터에 맞게 정규화한다."""
    doc = _raw_doc(
        "JPA N+1 문제를 EntityGraph와 fetch join으로 비교하고 해결했다. "
        "쿼리 수 차이를 로그로 확인했다.",
        meta={"topic": "  JPA   N+1 문제  "},
    )

    chunks = extract_evidence(doc)

    assert chunks[0].topic == "jpa n+1"


def test_extract_evidence_prefers_specific_topic_over_github_language() -> None:
    """GitHub 코드는 언어보다 본문에 드러난 구체 기술 topic을 우선한다."""
    doc = _raw_doc(
        """
public class UserService {
    public void issueToken() {
        // JWT access token 발급
    }
}
""",
        source_type="github",
        title="MINITCEN/MiniPrj-Bugbug UserService.java",
        meta={
            "doc_type": "code",
            "file_path": "bugbug/src/main/java/com/example/UserService.java",
            "language": "java",
            "ownership": "user_touched",
        },
    )

    chunks = extract_evidence(doc)

    assert chunks[0].topic == "jwt"
    assert chunks[0].doc_type == "code"
    assert chunks[0].confidence > 0.6


def test_extract_evidence_scores_sparse_template_lower_than_specific_note() -> None:
    """구체적인 구현/해결 내용이 있는 문서는 템플릿보다 confidence가 높다."""
    sparse = _raw_doc(
        "TODO 작성 예정 내용 없음 placeholder " * 8,
        title="빈 회고",
    )
    specific = _raw_doc(
        """
Spring Security JWT 인증 구현 중 Access Token 만료 시 401 응답이 반복되는 문제가 있었다.
원인은 Refresh Token 재발급 API와 인증 필터 예외 경로가 분리되지 않았기 때문이다.
해결을 위해 재발급 endpoint를 permitAll로 열고, JwtAuthenticationFilter에서 만료 예외를
별도 핸들러로 전달하도록 구현했다.
""",
        meta={"doc_type": "troubleshooting"},
    )

    sparse_chunk = extract_evidence(sparse)[0]
    specific_chunk = extract_evidence(specific)[0]

    assert sparse_chunk.confidence < specific_chunk.confidence
    assert specific_chunk.confidence >= 0.65


def test_extract_evidence_uses_llm_once_and_keeps_original_section_text() -> None:
    """LLM은 문서당 1회만 호출하고 선택된 섹션의 원문 전체를 chunk에 담는다."""

    class FakeStructuredLLM:
        """LLM 호출 횟수와 입력 preview를 기록하는 fake structured LLM."""

        def __init__(self) -> None:
            self.calls: list[list[tuple[str, str]]] = []

        def invoke(self, messages: list[tuple[str, str]]) -> EvidenceExtractionDecision:
            """두 번째 섹션만 면접 가치가 있다고 응답한다."""
            self.calls.append(messages)
            return EvidenceExtractionDecision(
                topic="Spring Security JWT",
                doc_type="troubleshooting",
                valuable_section_ids=["s2"],
                confidence=0.88,
            )

    doc = _raw_doc(
        """
# 목차
단순 소개입니다.

# Spring Security JWT 장애 대응
Access Token 만료 시 401 응답이 반복되는 문제가 있었다.
원인은 Refresh Token 재발급 API와 인증 필터 예외 경로가 분리되지 않았기 때문이다.
해결을 위해 재발급 endpoint를 permitAll로 열고 JwtAuthenticationFilter에서
만료 예외를 별도 핸들러로 전달하도록 구현했다.
이 문장은 preview 제한 이후에도 원문 section에는 보존되어야 한다.
""",
        meta={"doc_type": "retrospective", "week": 3},
    )
    fake_llm = FakeStructuredLLM()

    chunks = extract_evidence(doc, use_llm=True, structured_llm=fake_llm)

    assert len(fake_llm.calls) == 1
    assert len(chunks) == 1
    assert chunks[0].topic == "spring security"
    assert chunks[0].doc_type == "troubleshooting"
    assert 0.0 <= chunks[0].confidence <= 1.0
    assert "Access Token 만료" in chunks[0].text
    assert "원문 section에는 보존" in chunks[0].text
    assert "단순 소개" not in chunks[0].text


def test_extract_evidence_uses_section_topic_from_llm() -> None:
    """서로 다른 섹션은 각각의 검색 topic을 가져야 한다."""

    class FakeStructuredLLM:
        """섹션별 topic 결정을 반환하는 fake LLM."""

        def invoke(self, messages: list[tuple[str, str]]) -> EvidenceExtractionDecision:
            return EvidenceExtractionDecision(
                sections=[
                    EvidenceSectionDecision(
                        section_id="s1", topic="jwt", confidence=0.8
                    ),
                    EvidenceSectionDecision(
                        section_id="s2", topic="jpa", confidence=0.8
                    ),
                ]
            )

    doc = _raw_doc(
        """
# JWT 인증
Access Token 만료를 처리하기 위해 Refresh Token 재발급 API를 구현했다.

# JPA 조회 최적화
EntityGraph로 N+1 쿼리 문제를 줄이고 로그에서 쿼리 수를 확인했다.
""",
        meta={"doc_type": "retrospective"},
    )

    chunks = extract_evidence(doc, use_llm=True, structured_llm=FakeStructuredLLM())

    assert [chunk.topic for chunk in chunks] == ["jwt", "jpa"]


def test_extract_evidence_falls_back_when_llm_fails() -> None:
    """LLM 구조화 추출 실패 시 기존 규칙 기반 추출 결과를 반환한다."""

    class FailingStructuredLLM:
        """invoke에서 예외를 던지는 fake structured LLM."""

        def invoke(self, messages: list[tuple[str, str]]) -> EvidenceExtractionDecision:
            """LLM 장애를 흉내낸다."""
            raise RuntimeError("llm unavailable")

    doc = _raw_doc(
        "JPA N+1 문제를 EntityGraph와 fetch join으로 비교하고 해결했다. "
        "쿼리 수 차이를 로그로 확인했고 성능 개선 결과를 기록했다.",
        meta={"topic": "JPA N+1 문제", "doc_type": "troubleshooting"},
    )

    chunks = extract_evidence(doc, use_llm=True, structured_llm=FailingStructuredLLM())

    assert len(chunks) == 1
    assert chunks[0].topic == "jpa n+1"
    assert "EntityGraph" in chunks[0].text
