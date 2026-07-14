"""원본(RawDoc) -> 근거 추출 + 메타데이터 부여.

기술 개념, 코드 예시, 프로젝트 경험, 결정사항, 구현 근거를 골라내고
주제/주차/날짜/문서유형/신뢰도 메타데이터를 붙인다.
"""

import re
from dataclasses import dataclass
from hashlib import sha1
from typing import Protocol

from interview.llm.client import get_llm
from interview.evidence.sources import RawDoc
from interview.schemas.evidence import (
    EvidenceChunk,
    EvidenceExtractionDecision,
    EvidenceSectionCandidate,
)


@dataclass(frozen=True)
class _Section:
    """원문에서 잘라낸 면접 근거 후보 섹션."""

    section_id: str
    text: str
    heading: str | None = None


class StructuredEvidenceExtractor(Protocol):
    """EvidenceExtractionDecision을 반환하는 structured LLM 인터페이스."""

    def invoke(self, messages: list[tuple[str, str]]) -> EvidenceExtractionDecision:
        """LLM 메시지를 실행해 문서 단위 추출 결정을 반환한다."""


def extract_evidence(
    doc: RawDoc,
    *,
    use_llm: bool = False,
    structured_llm: StructuredEvidenceExtractor | None = None,
) -> list[EvidenceChunk]:
    """RawDoc 한 건에서 근거 후보들을 추출한다.

    여기서는 아직 임베딩하지 않는다. 텍스트 + 메타데이터까지만 만들고
    chunking 으로 넘긴다. 문서 단위 LLM 구조화 추출을 붙이더라도 이 함수의
    반환 계약은 유지하고, 실패 시 규칙 기반 결과를 사용할 수 있게 한다.
    """
    text = _clean_evidence_text(doc.raw_text)
    if not _has_interview_value(text, doc):
        return []

    if use_llm:
        try:
            chunks = _extract_with_llm(doc, text, structured_llm=structured_llm)
            if chunks:
                return chunks
        except Exception:
            pass

    return _extract_rule_based(doc, text)


def _extract_rule_based(doc: RawDoc, text: str) -> list[EvidenceChunk]:
    """규칙 기반으로 섹션별 주제와 신뢰도를 부여한다."""
    sections = _split_sections(text)
    if not sections:
        sections = [_Section(section_id="s1", text=text)]

    return [
        _new_chunk(
            doc,
            chunk_id=_section_chunk_id(doc, section.section_id),
            text=section.text,
            topic=_infer_topic(doc, section.text),
            confidence=_score_confidence(doc, section.text),
        )
        for section in sections
        if _has_interview_value(section.text, doc)
    ]


def _extract_with_llm(
    doc: RawDoc,
    text: str,
    *,
    structured_llm: StructuredEvidenceExtractor | None = None,
) -> list[EvidenceChunk]:
    """문서 1건당 LLM 1회로 면접 가치가 있는 섹션을 고른다.

    LLM에는 section preview만 전달하고, 반환된 section_id에 해당하는 원문
    섹션 전체를 EvidenceChunk.text로 사용한다.
    """
    sections = _split_sections(text)
    if not sections:
        return []

    decision = _decide_sections_with_llm(
        doc=doc,
        sections=[
            EvidenceSectionCandidate(
                section_id=section.section_id,
                heading=section.heading,
                preview=_preview(section.text),
                topic_candidates=_topic_candidates(doc, section.text),
            )
            for section in sections
        ],
        structured_llm=structured_llm,
    )

    decision_by_section = {item.section_id: item for item in decision.sections}
    selected_ids = set(decision.valuable_section_ids) | set(decision_by_section)
    selected_sections = [
        section
        for section in sections
        if section.section_id in selected_ids and _has_interview_value(section.text, doc)
    ]
    if not selected_sections:
        return []

    return [
        _new_chunk(
            doc,
            chunk_id=_section_chunk_id(doc, section.section_id),
            text=section.text,
            topic=_validated_llm_topic(
                decision_by_section.get(section.section_id).topic
                if section.section_id in decision_by_section
                else decision.topic,
                doc,
                section.text,
            ),
            confidence=_llm_confidence(
                decision_by_section.get(section.section_id).confidence
                if section.section_id in decision_by_section
                else decision.confidence,
                doc,
                section.text,
            ),
            doc_type=(
                decision_by_section[section.section_id].doc_type
                if section.section_id in decision_by_section
                else decision.doc_type
            )
            or doc.meta.get("doc_type"),
        )
        for section in selected_sections
    ]


def _decide_sections_with_llm(
    *,
    doc: RawDoc,
    sections: list[EvidenceSectionCandidate],
    structured_llm: StructuredEvidenceExtractor | None = None,
) -> EvidenceExtractionDecision:
    """문서 1건을 LLM에 한 번만 보내 면접 가치 있는 섹션을 선택한다."""
    if structured_llm is None:
        structured_llm = get_llm(temperature=0.0).with_structured_output(
            EvidenceExtractionDecision
        )

    section_payload = "\n\n".join(
        (
            f"[{section.section_id}]"
            f"\nheading: {section.heading or ''}"
            f"\ntopic_candidates: {', '.join(section.topic_candidates)}"
            f"\npreview:\n{section.preview}"
        )
        for section in sections
    )
    messages = [
        (
            "system",
            "\n".join(
                [
                    "당신은 개발자 면접용 근거 문서를 선별하는 도우미입니다.",
                    "RawDoc 한 건에서 면접 질문 생성에 가치 있는 섹션만 고릅니다.",
                    "목차, 빈 템플릿, 단순 체크리스트, 잡담은 제외합니다.",
                    "각 selected section마다 sections에 section_id, topic, confidence를 반환합니다.",
                    "topic은 제공한 topic_candidates 중 하나를 우선 사용합니다.",
                    "후보가 맞지 않을 때만 2~4단어의 일반적인 기술 주제를 사용합니다.",
                    "문서 전체에 하나의 topic을 재사용하지 말고 각 섹션 본문을 기준으로 판단합니다.",
                    "doc_type은 code, README, troubleshooting, retrospective, weekly_note, repository_meta 중 가장 가까운 값을 사용합니다.",
                    "반드시 제공된 section_id만 valuable_section_ids에 넣습니다.",
                ]
            ),
        ),
        (
            "human",
            "\n".join(
                [
                    f"title: {doc.title}",
                    f"source_type: {doc.source_type}",
                    f"source_url: {doc.source_url}",
                    f"meta: {doc.meta}",
                    "",
                    "sections:",
                    section_payload,
                ]
            ),
        ),
    ]
    return structured_llm.invoke(messages)


def _split_sections(text: str) -> list[_Section]:
    """Markdown heading/details/fenced code 경계를 우선해 원문 섹션 후보를 만든다."""
    blocks = _split_blocks_preserving_fences(text)
    sections: list[_Section] = []
    current: list[str] = []
    heading: str | None = None

    for block in blocks:
        stripped = block.strip()
        starts_new_section = bool(
            re.match(r"^#{1,6}\s+\S+", stripped)
            or re.match(r"^<summary>.*</summary>$", stripped, re.IGNORECASE | re.DOTALL)
        )

        if starts_new_section and current:
            sections.append(_make_section(len(sections), current, heading))
            current = []

        if starts_new_section:
            heading = _strip_heading(stripped)
        current.append(block)

    if current:
        sections.append(_make_section(len(sections), current, heading))

    if len(sections) == 1 and len(sections[0].text) > 2500:
        return _split_long_section(sections[0])
    return sections


def _split_blocks_preserving_fences(text: str) -> list[str]:
    """빈 줄로 나누되 fenced code block 내부는 하나의 block으로 유지한다."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue

        if not in_fence and not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue

        current.append(line)

    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def _make_section(index: int, blocks: list[str], heading: str | None) -> _Section:
    return _Section(
        section_id=f"s{index + 1}",
        heading=heading,
        text="\n\n".join(block.strip() for block in blocks if block.strip()).strip(),
    )


def _split_long_section(section: _Section) -> list[_Section]:
    """heading이 없는 긴 문서는 문단 묶음 단위로 LLM 선택 후보를 나눈다."""
    paragraphs = section.text.split("\n\n")
    sections: list[_Section] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if current and current_len + paragraph_len > 1800:
            sections.append(_make_section(len(sections), current, section.heading))
            current = []
            current_len = 0

        current.append(paragraph)
        current_len += paragraph_len

    if current:
        sections.append(_make_section(len(sections), current, section.heading))
    return sections


def _strip_heading(value: str) -> str:
    value = re.sub(r"^#{1,6}\s+", "", value).strip()
    value = re.sub(r"^<summary>|</summary>$", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\*\*", "", value).strip()
    return value


def _preview(text: str, max_chars: int = 900) -> str:
    """LLM 판단용 preview를 만든다. 실제 저장 텍스트는 원문 section 전체를 쓴다."""
    compact = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit("\n", 1)[0].strip() or compact[:max_chars]


MIN_TEXT_CHARS = 50
MIN_CODE_CHARS = 40
EXCLUDED_DOC_TYPES = {
    "directory_tree",
}

NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^<empty-block\s*/?>$", re.IGNORECASE),
    re.compile(r"^</?(ancestor-path|properties|content|page|database|data-sources)[^>]*>$", re.IGNORECASE),
    re.compile(r"^<parent-data-source\s+[^>]+/?>$", re.IGNORECASE),
    re.compile(r"^<ancestor-\d+-database\s+[^>]+/?>$", re.IGNORECASE),
    re.compile(r'^\{"date:[^"]+".*"url":"https://app\.notion\.com/p/[^"]+".*\}$'),
    re.compile(r"^here is the result of .+ as of .+:?$", re.IGNORECASE),
    re.compile(r"^the title of this .+ is:", re.IGNORECASE),
    re.compile(r"^you can use the .+ tool .+", re.IGNORECASE),
    re.compile(r"^목차$"),
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^- \[[ xX]?\]\s*(완료)?$"),
    re.compile(r"^-?\s*실행 결과\s*:?\s*$"),
]

TECH_KEYWORDS = [
    ("spring security", "spring security"),
    ("spring boot", "spring boot"),
    ("websocket", "websocket"),
    ("oauth", "oauth"),
    ("jwt", "jwt"),
    ("n+1", "jpa n+1"),
    ("jpa", "jpa"),
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
    raw_text = re.sub(
        r"📅?\*\*?이번 주 학습 현황.*?</table>",
        "",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned_lines: list[str] = []
    previous_blank = False

    for line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _remove_image_references(line)
        line = re.sub(r"</?(?:details|callout)[^>]*>", "", line, flags=re.IGNORECASE)
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


def _remove_image_references(line: str) -> str:
    """이미지는 면접 근거로 쓰지 않으므로 Markdown/S3 이미지 참조를 제거한다."""
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
    line = re.sub(r"https://prod-files-secure\.s3[^\s)<>]+", "", line)
    return line.rstrip()


def _is_noise_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in NOISE_LINE_PATTERNS)


def _is_markdown_toc_link(line: str) -> bool:
    return bool(re.match(r"^[-*]\s+\[[^\]]+\]\(#[^)]+\)$", line))


def _has_interview_value(text: str, doc: RawDoc) -> bool:
    """너무 짧거나 템플릿만 남은 문서를 제외한다."""
    if doc.meta.get("doc_type") in EXCLUDED_DOC_TYPES:
        return False

    if _is_github_download_status_text(text):
        return False

    if _is_image_only_result(text):
        return False

    min_chars = MIN_CODE_CHARS if doc.meta.get("doc_type") == "code" else MIN_TEXT_CHARS
    if len(text.strip()) < min_chars:
        return False

    alpha_or_korean = re.findall(r"[A-Za-z가-힣0-9]", text)
    return len(alpha_or_korean) >= min_chars // 2


def _is_github_download_status_text(text: str) -> bool:
    return text.strip().lower().startswith("successfully downloaded text file")


def _is_image_only_result(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip().lower()
    if not compact:
        return True
    image_markers = ("x-amz-", "prod-files-secure.s3", "image.png", "image.jpg")
    return compact in {"- 실행 결과", "실행 결과"} or (
        any(marker in compact for marker in image_markers)
        and len(re.findall(r"[A-Za-z가-힣0-9]", compact)) < 80
    )


def _infer_topic(doc: RawDoc, text: str) -> str:
    """RawDoc 메타데이터와 본문에서 검색 필터에 쓸 topic을 추정한다."""
    candidates = [
        doc.meta.get("topic"),
        _topic_from_keywords(f"{doc.title}\n{text}"),
        _topic_from_github_meta(doc),
        doc.title,
    ]

    for candidate in candidates:
        topic = _normalize_topic(candidate)
        if topic:
            return topic

    return "general"


def _topic_candidates(doc: RawDoc, text: str) -> list[str]:
    """LLM이 임의의 topic을 만들지 않도록 섹션별 후보를 제공한다."""
    candidates: list[str] = []
    for candidate in (
        doc.meta.get("topic"),
        _topic_from_keywords(f"{doc.title}\n{text}"),
        _topic_from_github_meta(doc),
    ):
        normalized = _normalize_topic(candidate)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates or ["general"]


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


def _validated_llm_topic(value: object, doc: RawDoc, text: str) -> str:
    """LLM topic을 정규화하고 섹션 본문과 무관한 값은 규칙 기반 값으로 바꾼다."""
    topic = _normalize_topic(value)
    candidates = _topic_candidates(doc, text)
    if topic and (
        topic in candidates
        or topic == "general"
        or _topic_from_keywords(topic) == topic
    ):
        return topic
    return _infer_topic(doc, text)


def _llm_confidence(value: object, doc: RawDoc, text: str) -> float:
    """LLM confidence가 있어도 규칙 점수와 큰 차이가 나지 않게 보정한다."""
    rule_score = _score_confidence(doc, text)
    try:
        llm_score = _clamp(float(value))
    except (TypeError, ValueError):
        return rule_score
    return _clamp((rule_score * 0.6) + (llm_score * 0.4))


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

    score = 0.32
    length = len(text)

    if length >= 150:
        score += 0.06
    if length >= 500:
        score += 0.06
    if length >= 1000:
        score += 0.05
    if length >= 3000:
        score += 0.04

    lower_text = text.lower()
    doc_type = doc.meta.get("doc_type")
    if doc_type in {"troubleshooting", "retrospective"}:
        score += 0.06
    if doc_type == "code":
        score += 0.12
    concrete_markers = ("error", "exception", "troubleshooting", "원인", "해결", "구현", "설계")
    marker_count = sum(1 for token in concrete_markers if token in lower_text)
    score += min(marker_count * 0.04, 0.12)
    if "원인" in lower_text and "해결" in lower_text:
        score += 0.10
    if re.search(r"\b(class|def|function|public|private|select|insert|update)\b", lower_text):
        score += 0.06
    if doc.meta.get("ownership") == "user_touched":
        score += 0.18
    elif doc.source_type == "github" and doc_type == "code":
        score -= 0.05

    if doc.meta.get("language") in {"html", "css", "scss", "sql"}:
        score -= 0.10

    if length < 80 and doc_type != "code":
        score -= 0.15
    if _looks_like_template(text):
        score -= 0.20

    return _clamp(min(score, 0.92))


def _new_chunk(
    doc: RawDoc,
    *,
    chunk_id: str,
    text: str,
    topic: str,
    confidence: float,
    doc_type: str | None = None,
) -> EvidenceChunk:
    """RawDoc 메타데이터를 보존한 EvidenceChunk를 생성한다."""
    commit_shas = doc.meta.get("commit_shas")
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        source_type=doc.source_type,
        source_url=doc.source_url,
        topic=topic,
        doc_type=doc_type or doc.meta.get("doc_type"),
        week=doc.meta.get("week"),
        date=doc.meta.get("date"),
        confidence=confidence,
        file_path=doc.meta.get("file_path"),
        language=doc.meta.get("language"),
        ownership=doc.meta.get("ownership"),
        commit_count=int(doc.meta.get("commit_count") or 0),
        last_commit_sha=doc.meta.get("last_commit_sha")
        or (commit_shas[0] if isinstance(commit_shas, list) and commit_shas else None),
    )


def refine_evidence_chunks(chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
    """청킹 뒤 자식 텍스트 기준으로 topic과 confidence를 다시 계산한다."""
    refined: list[EvidenceChunk] = []
    for chunk in chunks:
        doc = RawDoc(
            source_url=chunk.source_url,
            source_type=chunk.source_type.value,
            title=chunk.file_path or chunk.source_url,
            raw_text=chunk.text,
            meta={
                "topic": None,
                "doc_type": chunk.doc_type,
                "language": chunk.language,
                "ownership": chunk.ownership,
                "commit_count": chunk.commit_count,
            },
        )
        refined.append(
            chunk.model_copy(
                update={
                    # LLM/섹션 추출 단계가 이미 정한 topic은 보존한다. 코드가
                    # max_chars로 잘린 뒤에도 앞 청크의 topic으로 덮어쓰지 않는다.
                    "topic": _normalize_topic(chunk.topic) or _infer_topic(doc, chunk.text),
                    "confidence": _score_confidence(doc, chunk.text),
                }
            )
        )
    return refined


def _looks_like_template(text: str) -> bool:
    lower_text = text.lower()
    template_markers = (
        "todo",
        "작성 예정",
        "내용 없음",
        "입력",
        "placeholder",
        "successfully downloaded text file",
    )
    return any(marker in lower_text for marker in template_markers)


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _chunk_id(doc: RawDoc) -> str:
    """원본 문서의 출처 정보를 바탕으로 안정적인 기본 chunk id 를 만든다."""
    digest = sha1(f"{doc.source_type}:{doc.source_url}:{doc.title}".encode()).hexdigest()
    return digest[:12]


def _section_chunk_id(doc: RawDoc, section_id: str) -> str:
    """원본 문서와 LLM 선택 섹션 ID를 바탕으로 안정적인 chunk id 를 만든다."""
    digest = sha1(
        f"{doc.source_type}:{doc.source_url}:{doc.title}:{section_id}".encode()
    ).hexdigest()
    return digest[:12]
