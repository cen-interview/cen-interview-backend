"""로컬 최종 Parquet을 PostgreSQL pgvector 테이블에 적재한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from interview.config import settings
from interview.evidence.question_patterns import get_question_pattern_store


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--embedding-text-mode", default="pattern_with_signals")
    parser.add_argument("--embedding-text-version", default="1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(
    path: Path,
    dataset_version: str,
    *,
    embedding_text_mode: str = "pattern_with_signals",
    embedding_text_version: str = "1",
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    required = {"pattern_id", "pattern_text", "variants", "frequency", "signal_kind", "required_evidence_signals", "topic_family", "embedding", "embedding_model", "embedding_dimensions", "embedding_text_mode", "embedding_text_version", "dataset_version"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"적재 입력 필드가 없습니다: {missing}")
    models = set(table.column("embedding_model").to_pylist())
    dimensions = set(table.column("embedding_dimensions").to_pylist())
    if models != {settings.embedding_model}:
        raise ValueError(
            f"임베딩 모델이 서버 설정과 다릅니다: file={sorted(models)}, expected={settings.embedding_model}"
        )
    if dimensions != {settings.embedding_dimensions}:
        raise ValueError(
            f"임베딩 차원이 서버 설정과 다릅니다: file={sorted(dimensions)}, expected={settings.embedding_dimensions}"
        )
    modes = set(table.column("embedding_text_mode").to_pylist())
    versions = set(table.column("embedding_text_version").to_pylist())
    if modes != {embedding_text_mode}:
        raise ValueError(
            f"임베딩 텍스트 모드가 다릅니다: file={sorted(modes)}, expected={embedding_text_mode}"
        )
    if versions != {embedding_text_version}:
        raise ValueError(
            f"임베딩 텍스트 버전이 다릅니다: file={sorted(versions)}, expected={embedding_text_version}"
        )
    dataset_versions = set(table.column("dataset_version").to_pylist())
    if dataset_versions != {dataset_version}:
        raise ValueError(f"dataset_version이 입력 전체에서 일치하지 않습니다: {dataset_versions}")
    rows = table.to_pylist()
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        pattern_id = str(row["pattern_id"])
        if pattern_id in seen:
            raise ValueError(f"pattern_id가 중복됩니다: {pattern_id}")
        seen.add(pattern_id)
        if str(row["dataset_version"]) != dataset_version:
            raise ValueError(f"dataset_version이 일치하지 않습니다: {pattern_id}")
        embedding = row["embedding"]
        if embedding is None or len(embedding) != settings.embedding_dimensions:
            raise ValueError(f"임베딩 차원이 설정과 다릅니다: {pattern_id}")
        result.append(
            {
                "pattern_id": pattern_id,
                "pattern_text": str(row["pattern_text"]).strip(),
                "variants": row["variants"] or [],
                "frequency": int(row["frequency"]),
                "signal_kind": str(row["signal_kind"]),
                "required_evidence_signals": row["required_evidence_signals"] or [],
                "topic_family": row["topic_family"],
                "embedding": embedding,
                "dataset_version": dataset_version,
                "updated_at": datetime.utcnow(),
            }
        )
    return result


def ingest(
    path: Path,
    dataset_version: str,
    *,
    batch_size: int = 500,
    dry_run: bool = False,
    embedding_text_mode: str = "pattern_with_signals",
    embedding_text_version: str = "1",
) -> dict[str, Any]:
    rows = load_rows(
        path,
        dataset_version,
        embedding_text_mode=embedding_text_mode,
        embedding_text_version=embedding_text_version,
    )
    report: dict[str, Any] = {
        "status": "dry_run" if dry_run else "running",
        "input": str(path),
        "input_sha256": _checksum(path),
        "dataset_version": dataset_version,
        "rows": len(rows),
        "database_backend": "postgresql-pgvector",
        "embedding_text_mode": embedding_text_mode,
        "embedding_text_version": embedding_text_version,
    }
    if not dry_run:
        report["upserted_rows"] = get_question_pattern_store().replace_dataset(
            rows, dataset_version, batch_size
        )
        report["status"] = "success"
    return report


def main() -> None:
    args = _args()
    report = ingest(args.input, args.dataset_version, batch_size=args.batch_size, dry_run=args.dry_run, embedding_text_mode=args.embedding_text_mode, embedding_text_version=args.embedding_text_version)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
