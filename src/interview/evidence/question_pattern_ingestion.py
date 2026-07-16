"""배포된 CSV 질문 패턴을 PostgreSQL에 적재한다."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from interview.config import settings
from interview.evidence.question_patterns import get_question_pattern_store

_REQUIRED_COLUMNS = {
    "pattern_id",
    "pattern_text",
    "frequency",
    "signal_kind",
    "required_evidence_signals",
    "topic_family",
    "embedding",
    "dataset_version",
    "variants",
}


def _json_list(value: str, *, column: str, pattern_id: str) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{column} 값이 올바른 JSON 배열이 아닙니다: {pattern_id}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{column} 값은 배열이어야 합니다: {pattern_id}")
    return parsed


def load_question_pattern_rows(csv_path: Path) -> tuple[list[dict[str, Any]], str]:
    """CSV를 읽어 질문 패턴 저장소가 받는 행과 데이터셋 버전을 반환한다."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = sorted(_REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"질문 패턴 CSV 필수 컬럼이 없습니다: {missing}")

        rows: list[dict[str, Any]] = []
        pattern_ids: set[str] = set()
        dataset_versions: set[str] = set()
        now = datetime.now(timezone.utc)

        for line_number, source_row in enumerate(reader, start=2):
            pattern_id = (source_row["pattern_id"] or "").strip()
            pattern_text = (source_row["pattern_text"] or "").strip()
            dataset_version = (source_row["dataset_version"] or "").strip()
            if not pattern_id or not pattern_text or not dataset_version:
                raise ValueError(f"필수 값이 비어 있습니다: line={line_number}")
            if pattern_id in pattern_ids:
                raise ValueError(f"pattern_id가 중복됩니다: {pattern_id}")

            embedding = _json_list(
                source_row["embedding"],
                column="embedding",
                pattern_id=pattern_id,
            )
            if len(embedding) != settings.embedding_dimensions:
                raise ValueError(
                    "embedding 차원이 설정과 다릅니다: "
                    f"pattern_id={pattern_id}, actual={len(embedding)}, "
                    f"expected={settings.embedding_dimensions}"
                )

            pattern_ids.add(pattern_id)
            dataset_versions.add(dataset_version)
            rows.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_text": pattern_text,
                    "frequency": int(source_row["frequency"]),
                    "signal_kind": (source_row["signal_kind"] or "").strip(),
                    "required_evidence_signals": _json_list(
                        source_row["required_evidence_signals"],
                        column="required_evidence_signals",
                        pattern_id=pattern_id,
                    ),
                    "topic_family": (source_row["topic_family"] or "").strip() or None,
                    "embedding": [float(value) for value in embedding],
                    "dataset_version": dataset_version,
                    "variants": _json_list(
                        source_row["variants"],
                        column="variants",
                        pattern_id=pattern_id,
                    ),
                    "updated_at": now,
                }
            )

    if not rows:
        raise ValueError("질문 패턴 CSV에 데이터가 없습니다.")
    if len(dataset_versions) != 1:
        raise ValueError(f"dataset_version이 하나로 일치하지 않습니다: {sorted(dataset_versions)}")
    return rows, next(iter(dataset_versions))


def ingest_question_patterns(csv_path: Path) -> int:
    """CSV가 표현하는 질문 패턴 데이터셋으로 DB 내용을 교체한다."""
    rows, dataset_version = load_question_pattern_rows(csv_path)
    return get_question_pattern_store().replace_dataset(rows, dataset_version)
