"""원본(RawDoc) -> 근거 추출 + 메타데이터 부여.

기술 개념, 코드 예시, 프로젝트 경험, 결정사항, 구현 근거를 골라내고
주제/주차/날짜/문서유형/신뢰도 메타데이터를 붙인다.
"""

import re
from hashlib import sha1

from interview.evidence.sources import RawDoc
from interview.schemas.evidence import EvidenceChunk


def extract_evidence(doc: RawDoc) -> list[EvidenceChunk]:
    """RawDoc 한 건에서 근거 후보들을 추출한다.

    여기서는 아직 임베딩하지 않는다. 텍스트 + 메타데이터까지만 만들고
    chunking 으로 넘긴다. 문서 단위 LLM 구조화 추출을 붙이더라도 이 함수의
    반환 계약은 유지하고, 실패 시 규칙 기반 결과를 사용할 수 있게 한다.
    """
    text = _clean_evidence_text(doc.raw_text)
    if not _has_interview_value(text, doc):
        return []

    topic = _infer_topic(doc, text)
    confidence = _score_confidence(doc, text)

    return [
        EvidenceChunk(
            chunk_id=_chunk_id(doc),
            text=text,
            source_type=doc.source_type,
            source_url=doc.source_url,
            topic=topic,
            doc_type=doc.meta.get("doc_type"),
            week=doc.meta.get("week"),
            date=doc.meta.get("date"),
            confidence=confidence,
        )
    ]


MIN_TEXT_CHARS = 50
MIN_CODE_CHARS = 40

NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^<empty-block\s*/?>$", re.IGNORECASE),
    re.compile(r"^</?(ancestor-path|properties|content|page|database|data-sources)[^>]*>$", re.IGNORECASE),
    re.compile(r"^here is the result of .+ as of .+:?$", re.IGNORECASE),
    re.compile(r"^the title of this .+ is:", re.IGNORECASE),
    re.compile(r"^you can use the .+ tool .+", re.IGNORECASE),
    re.compile(r"^목차$"),
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^- \[[ xX]?\]\s*(완료)?$"),
]

TECH_KEYWORDS = [
    ("spring security", "spring security"),
    ("spring boot", "spring boot"),
    ("websocket", "websocket"),
    ("oauth", "oauth"),
    ("jwt", "jwt"),
    ("jpa", "jpa"),
    ("n+1", "jpa n+1"),
    ("redis", "redis"),
    ("docker", "docker"),
    ("kubernetes", "kubernetes"),
    ("sql", "sql"),
    ("mysql", "mysql"),
    ("postgres", "postgresql"),
    ("java", "java"),
    ("python", "python"),
    ("typescript", "typescript"),
    ("javascript", "javascript"),
    ("rag", "rag"),
    ("mcp", "mcp"),
    ("troubleshooting", "troubleshooting"),
    ("트러블슈팅", "troubleshooting"),
    ("인증", "auth"),
    ("시큐리티", "spring security"),
    ("웹소켓", "websocket"),
]


def _clean_evidence_text(raw_text: str) -> str:
    """목차, 빈 블록, MCP boilerplate, 반복 체크리스트를 제거한다."""
    cleaned_lines: list[str] = []
    previous_blank = False

    for line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if _is_noise_line(stripped):
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
                previous_blank = True
            continue

        if _is_markdown_toc_link(stripped):
            continue

        cleaned_lines.append(line.rstrip())
        previous_blank = False

    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _is_noise_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in NOISE_LINE_PATTERNS)


def _is_markdown_toc_link(line: str) -> bool:
    return bool(re.match(r"^[-*]\s+\[[^\]]+\]\(#[^)]+\)$", line))


def _has_interview_value(text: str, doc: RawDoc) -> bool:
    """너무 짧거나 템플릿만 남은 문서를 제외한다."""
    min_chars = MIN_CODE_CHARS if doc.meta.get("doc_type") == "code" else MIN_TEXT_CHARS
    if len(text.strip()) < min_chars:
        return False

    alpha_or_korean = re.findall(r"[A-Za-z가-힣0-9]", text)
    return len(alpha_or_korean) >= min_chars // 2


def _infer_topic(doc: RawDoc, text: str) -> str:
    """RawDoc 메타데이터와 본문에서 검색 필터에 쓸 topic을 추정한다."""
    candidates = [
        doc.meta.get("topic"),
        _topic_from_github_meta(doc),
        _topic_from_keywords(f"{doc.title}\n{text}"),
        doc.title,
    ]

    for candidate in candidates:
        topic = _normalize_topic(candidate)
        if topic:
            return topic

    return "general"


def _topic_from_github_meta(doc: RawDoc) -> str | None:
    if doc.source_type != "github":
        return None

    file_path = str(doc.meta.get("file_path") or "")
    language = str(doc.meta.get("language") or "")
    lower_path = file_path.lower()

    for keyword, topic in TECH_KEYWORDS:
        if keyword in lower_path:
            return topic
    if language:
        return language
    return None


def _topic_from_keywords(text: str) -> str | None:
    lower_text = text.lower()
    for keyword, topic in TECH_KEYWORDS:
        if keyword in lower_text:
            return topic
    return None


def _normalize_topic(value: object) -> str | None:
    """자유 문자열 topic을 최소한의 일관된 표기로 정규화한다."""
    if not isinstance(value, str):
        return None

    topic = value.strip().lower()
    if not topic:
        return None

    topic = re.sub(r"[_\-]+", " ", topic)
    topic = re.sub(r"\s+", " ", topic)
    topic = topic.strip(" #[](){}")

    aliases = {
        "springsecurity": "spring security",
        "spring security jwt": "spring security",
        "jwt token": "jwt",
        "jpa n 1": "jpa n+1",
        "jpa n+1 문제": "jpa n+1",
        "postgres": "postgresql",
    }
    topic = aliases.get(topic, topic)

    if len(topic) > 60:
        topic = topic[:60].rsplit(" ", 1)[0] or topic[:60]
    return topic or None


def _score_confidence(doc: RawDoc, text: str) -> float:
    """내용 길이와 구체성 기반으로 근거 신뢰도를 산정한다."""
    explicit = doc.meta.get("confidence")
    if explicit is not None:
        try:
            return _clamp(float(explicit))
        except (TypeError, ValueError):
            pass

    score = 0.40
    length = len(text)

    if length >= 150:
        score += 0.10
    if length >= 500:
        score += 0.10
    if length >= 1000:
        score += 0.10
    if length >= 3000:
        score += 0.10

    lower_text = text.lower()
    doc_type = doc.meta.get("doc_type")
    if doc_type in {"code", "troubleshooting", "retrospective"}:
        score += 0.08
    if "```" in text or doc.meta.get("doc_type") == "code":
        score += 0.12
    concrete_markers = ("error", "exception", "troubleshooting", "원인", "해결", "구현", "설계")
    marker_count = sum(1 for token in concrete_markers if token in lower_text)
    score += min(marker_count * 0.06, 0.18)
    if re.search(r"\b(class|def|function|public|private|select|insert|update)\b", lower_text):
        score += 0.08
    if doc.meta.get("date") or doc.meta.get("week") is not None:
        score += 0.05
    if doc.meta.get("ownership") == "user_touched":
        score += 0.05

    if length < 80 and doc_type != "code":
        score -= 0.15
    if _looks_like_template(text):
        score -= 0.20

    return _clamp(score)


def _looks_like_template(text: str) -> bool:
    lower_text = text.lower()
    template_markers = ("todo", "작성 예정", "내용 없음", "입력", "placeholder")
    return any(marker in lower_text for marker in template_markers)


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _chunk_id(doc: RawDoc) -> str:
    """원본 문서의 출처 정보를 바탕으로 안정적인 기본 chunk id 를 만든다."""
    digest = sha1(f"{doc.source_type}:{doc.source_url}:{doc.title}".encode()).hexdigest()
    return digest[:12]
