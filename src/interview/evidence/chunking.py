"""근거 청킹.

추출된 EvidenceChunk 텍스트가 너무 길면 임베딩에 적합한 크기로 쪼갠다.
메타데이터는 자식 청크에 그대로 상속한다.
"""

from interview.schemas.evidence import EvidenceChunk


def chunk(chunks: list[EvidenceChunk], max_chars: int = 1000) -> list[EvidenceChunk]:
    """긴 청크를 max_chars 기준으로 분할.

    TODO(담당 A):
      - 문단/코드 경계를 존중하는 분할 (문장 중간 자르지 않기)
      - 분할된 자식에 새 chunk_id 부여, 메타데이터 상속
    """
    result: list[EvidenceChunk] = []

    for item in chunks:
        text = item.text.strip()
        if len(text) <= max_chars:
            result.append(item.model_copy(update={"text": text}))
            continue

        parts = _split_text(text, max_chars)
        for index, part in enumerate(parts):
            result.append(
                item.model_copy(
                    update={
                        "chunk_id": f"{item.chunk_id}:{index}",
                        "text": part,
                    }
                )
            )

    return result


def _split_text(text: str, max_chars: int) -> list[str]:
    """문단 우선으로 나누고, 너무 긴 문단만 hard split 한다."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parts: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(
                paragraph[start : start + max_chars].strip()
                for start in range(0, len(paragraph), max_chars)
            )
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = paragraph

    if current:
        parts.append(current)

    return parts
