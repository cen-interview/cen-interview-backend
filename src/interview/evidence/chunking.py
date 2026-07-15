"""근거 청킹.

추출된 EvidenceChunk 텍스트가 너무 길면 임베딩에 적합한 크기로 쪼갠다.
메타데이터는 자식 청크에 그대로 상속한다.
"""

import re

from interview.evidence.code_chunking import normalize_code_text, split_code_units
from interview.schemas.evidence import EvidenceChunk


def chunk(chunks: list[EvidenceChunk], max_chars: int = 1000) -> list[EvidenceChunk]:
    """긴 청크를 max_chars 기준으로 분할한다.

    일반 문서는 문단 경계를 우선하고, 코드 문서는 code fence 또는 빈 줄
    경계를 우선한다. 분할된 자식 청크는 ``{parent_id}-{n}`` 규칙을 따른다.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    result: list[EvidenceChunk] = []
    for source_chunk in chunks:
        text = (
            normalize_code_text(source_chunk.text)
            if _is_code_chunk(source_chunk)
            else source_chunk.text.strip()
        )
        if not text:
            continue
        if len(text) <= max_chars:
            result.append(source_chunk.model_copy(update={"text": text}))
            continue

        parts = (
            _split_code_text(text, source_chunk.language, max_chars)
            if _is_code_chunk(source_chunk)
            else _split_text(text, max_chars)
        )
        if len(parts) == 1:
            result.append(source_chunk.model_copy(update={"text": parts[0]}))
            continue

        for index, part in enumerate(parts, start=1):
            result.append(
                source_chunk.model_copy(
                    update={
                        "chunk_id": f"{source_chunk.chunk_id}-{index}",
                        "text": part,
                    }
                )
            )

    return result


def _is_code_chunk(chunk: EvidenceChunk) -> bool:
    """코드 전용 분할 규칙을 적용할 청크인지 판단한다."""
    if chunk.doc_type == "code":
        return True
    text = chunk.text.strip()
    return text.startswith("```") or "\n```" in text


def _split_text(text: str, max_chars: int) -> list[str]:
    """문단 경계를 우선해 일반 텍스트를 분할한다."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
    if not blocks:
        return []

    parts: list[str] = []
    current = ""
    for block in blocks:
        for unit in _split_oversized_block(block, max_chars):
            current = _append_unit(parts, current, unit, max_chars, separator="\n\n")

    if current:
        parts.append(current.strip())
    return parts


def _split_code_text(text: str, language: str | None, max_chars: int) -> list[str]:
    """코드 문서를 code fence 또는 빈 줄 경계 기준으로 분할한다."""
    if "```" not in text:
        return _pack_code_units(split_code_units(text, language, max_chars), max_chars)

    blocks = _split_code_blocks(text)
    parts: list[str] = []
    current = ""

    for block in blocks:
        for unit in _split_oversized_code_block(block, max_chars):
            current = _append_unit(parts, current, unit, max_chars, separator="\n\n")

    if current:
        parts.append(current.strip())
    return parts


def _pack_code_units(units: list[str], max_chars: int) -> list[str]:
    """작은 선언 단위는 함께 묶되 선언 경계를 넘겨 중간에서 자르지 않는다."""
    parts: list[str] = []
    current = ""
    for unit in units:
        current = _append_unit(parts, current, unit, max_chars, separator="\n\n")
    if current:
        parts.append(current.strip())
    return parts


def _split_code_blocks(text: str) -> list[str]:
    """fenced code block은 하나의 단위로 유지하고 나머지는 빈 줄 기준으로 나눈다."""
    if "```" not in text:
        return [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]

    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    for line in text.strip().splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and not line.strip() and current:
            blocks.append("\n".join(current).strip())
            current = []
            continue
        current.append(line.rstrip())

    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def _append_unit(
    parts: list[str],
    current: str,
    unit: str,
    max_chars: int,
    *,
    separator: str,
) -> str:
    """현재 청크에 unit을 붙이거나, 넘치면 현재 청크를 확정한다."""
    unit = unit.strip()
    if not unit:
        return current
    if not current:
        return unit

    candidate = f"{current}{separator}{unit}"
    if len(candidate) <= max_chars:
        return candidate

    parts.append(current.strip())
    return unit


def _split_oversized_block(block: str, max_chars: int) -> list[str]:
    """한 문단이 max_chars를 넘으면 문장 경계 기준으로 더 나눈다."""
    if len(block) <= max_chars:
        return [block.strip()]

    sentences = _split_sentences(block)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                parts.append(current.strip())
                current = ""
            parts.extend(_hard_split(sentence, max_chars))
            continue
        current = _append_unit(parts, current, sentence, max_chars, separator=" ")

    if current:
        parts.append(current.strip())
    return parts


def _split_oversized_code_block(block: str, max_chars: int) -> list[str]:
    """긴 코드 블록은 줄 경계 기준으로 나눈다."""
    if len(block) <= max_chars:
        return [block.strip()]

    parts: list[str] = []
    current = ""
    for line in block.splitlines():
        if len(line) > max_chars:
            if current:
                parts.append(current.rstrip())
                current = ""
            parts.extend(_hard_split(line, max_chars))
            continue

        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current.rstrip())
            current = line

    if current:
        parts.append(current.rstrip())
    return [part.strip() for part in parts if part.strip()]


def _split_sentences(text: str) -> list[str]:
    """문장 끝 기호 뒤 공백을 기준으로 문장을 나눈다."""
    pieces = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """경계가 없는 긴 문자열을 마지막 수단으로 고정 길이에 가깝게 나눈다."""
    return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars)]
