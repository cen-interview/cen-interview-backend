"""소스 코드의 의미 단위를 보존하는 청킹 보조 함수."""

from __future__ import annotations

import ast
import re


_DECLARATION_PATTERNS = {
    "java": re.compile(r"^\s*(?:(?:public|protected|private|static|final|abstract|synchronized)\s+)*(?:class|interface|enum|record|[\w<>\[\], ?]+\s+\w+)\s*\([^;]*\)\s*\{|^\s*(?:public|protected|private)?\s*(?:class|interface|enum|record)\s+"),
    "kotlin": re.compile(r"^\s*(?:class|object|interface|fun)\s+"),
    "javascript": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+|^\s*(?:export\s+)?class\s+|^\s*(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"),
    "typescript": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+|^\s*(?:export\s+)?(?:class|interface|type)\s+|^\s*(?:export\s+)?(?:const|let)\s+\w+\s*="),
}


def normalize_code_text(text: str) -> str:
    """코드 들여쓰기는 보존하면서 불필요한 공백만 정리한다."""
    lines = [line.rstrip() for line in text.strip().splitlines()]
    result: list[str] = []
    blank_count = 0
    for line in lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= 1:
                result.append("")
            continue
        blank_count = 0
        result.append(line)
    return "\n".join(result).strip()


def split_code_units(text: str, language: str | None, max_chars: int) -> list[str]:
    """함수·클래스 선언을 우선해 코드 단위를 만들고 큰 단위만 줄 기준으로 나눈다."""
    normalized = normalize_code_text(text)
    if not normalized:
        return []

    if language == "python":
        units = _python_units(normalized)
    else:
        units = _brace_units(normalized, language)

    if not units:
        units = [normalized]
    return [part for unit in units for part in _split_large_unit(unit, max_chars) if part]


def _python_units(text: str) -> list[str]:
    """Python AST의 lineno/end_lineno를 이용해 모듈의 선언 단위를 보존한다."""
    try:
        module = ast.parse(text)
    except SyntaxError:
        return _blank_line_units(text)

    nodes = [
        node
        for node in module.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not nodes:
        return _blank_line_units(text)

    lines = text.splitlines()
    units: list[str] = []
    cursor = 0
    for node in nodes:
        start = _include_leading_python_comments(lines, node.lineno - 1, cursor)
        end = getattr(node, "end_lineno", node.lineno)
        if start > cursor:
            prefix = "\n".join(lines[cursor:start]).strip()
            if prefix:
                units.append(prefix)
        unit = "\n".join(lines[start:end]).strip()
        if unit:
            units.append(unit)
        cursor = end
    suffix = "\n".join(lines[cursor:]).strip()
    if suffix:
        units.append(suffix)
    return units


def _include_leading_python_comments(lines: list[str], start: int, lower_bound: int) -> int:
    """선언 직전의 연속된 주석과 doc 설명을 선언 청크에 포함한다."""
    index = start - 1
    while index >= lower_bound:
        stripped = lines[index].strip()
        if stripped.startswith("#") or not stripped:
            index -= 1
            continue
        break
    return index + 1


def _brace_units(text: str, language: str | None) -> list[str]:
    """중괄호 언어의 최상위 선언을 함수/클래스 단위로 분리한다."""
    pattern = _DECLARATION_PATTERNS.get(language or "")
    if pattern is None:
        return _blank_line_units(text)

    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if pattern.search(line)]
    if not starts:
        return _blank_line_units(text)

    units: list[str] = []
    cursor = 0
    for start in starts:
        if start < cursor:
            continue
        declaration_start = _include_leading_slash_comments(lines, start, cursor)
        if declaration_start > cursor:
            prefix = "\n".join(lines[cursor:declaration_start]).strip()
            if prefix:
                units.append(prefix)

        end = _find_brace_block_end(lines, start)
        if end <= start:
            end = start + 1
        unit = "\n".join(lines[declaration_start:end]).strip()
        if unit:
            units.append(unit)
        cursor = end
    suffix = "\n".join(lines[cursor:]).strip()
    if suffix:
        units.append(suffix)
    return units


def _include_leading_slash_comments(lines: list[str], start: int, lower_bound: int) -> int:
    index = start - 1
    while index >= lower_bound:
        stripped = lines[index].strip()
        if stripped.startswith(("//", "/*", "*", "*/")) or not stripped:
            index -= 1
            continue
        break
    return index + 1


def _find_brace_block_end(lines: list[str], start: int) -> int:
    depth = 0
    opened = False
    for index in range(start, len(lines)):
        line = re.sub(r"//.*$", "", lines[index])
        depth += line.count("{") - line.count("}")
        opened = opened or "{" in line
        if opened and depth <= 0:
            return index + 1
    return len(lines)


def _blank_line_units(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _split_large_unit(unit: str, max_chars: int) -> list[str]:
    if len(unit) <= max_chars:
        return [unit]

    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in unit.splitlines():
        line_size = len(line) + (1 if current else 0)
        if current and size + line_size > max_chars:
            parts.append("\n".join(current).strip())
            current, size = [], 0
        current.append(line)
        size += line_size
    if current:
        parts.append("\n".join(current).strip())
    return parts
