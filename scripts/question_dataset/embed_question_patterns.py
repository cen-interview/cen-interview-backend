"""최종 질문 패턴 Parquet을 지정한 문서 구성으로 임베딩한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from interview.config import settings

Embedder = Callable[[list[str]], list[list[float]]]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="실제 임베딩 API 호출을 명시적으로 승인한다.",
    )
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--dimensions", type=int, default=settings.embedding_dimensions)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--embedding-text-mode",
        choices=("pattern_only", "pattern_with_signals"),
        default="pattern_only",
    )
    parser.add_argument("--embedding-text-version", default="1")
    parser.add_argument("--dataset-version")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embed_patterns(
    input_path: Path,
    output_path: Path,
    *,
    batch_size: int = 128,
    limit: int | None = None,
    dry_run: bool = False,
    model: str = settings.embedding_model,
    dimensions: int = settings.embedding_dimensions,
    embedder: Embedder | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    embedding_text_mode: str = "pattern_only",
    embedding_text_version: str = "1",
    dataset_version_override: str | None = None,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if batch_size <= 0 or dimensions <= 0:
        raise ValueError("batch-size와 dimensions는 1 이상이어야 합니다.")
    table = pq.read_table(input_path)
    required = {"pattern_id", "pattern_text", "frequency", "signal_kind", "dataset_version"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"패턴 입력 필드가 없습니다: {missing}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit은 1 이상이어야 합니다.")
        table = table.slice(0, min(limit, table.num_rows))
    if embedding_text_mode not in {"pattern_only", "pattern_with_signals"}:
        raise ValueError(f"지원하지 않는 embedding_text_mode입니다: {embedding_text_mode}")
    ids = table.column("pattern_id").to_pylist()
    pattern_texts = table.column("pattern_text").to_pylist()
    signal_kinds = table.column("signal_kind").to_pylist()
    topic_families = table.column("topic_family").to_pylist()
    evidence_signals = table.column("required_evidence_signals").to_pylist()
    if embedding_text_mode == "pattern_with_signals":
        texts = [
            "\n".join(
                [
                    f"패턴: {pattern_text}",
                    f"종류: {signal_kind}",
                    f"주제: {topic_family or ''}",
                    f"필요 근거: {', '.join(evidence or [])}",
                ]
            )
            for pattern_text, signal_kind, topic_family, evidence in zip(
                pattern_texts,
                signal_kinds,
                topic_families,
                evidence_signals,
                strict=True,
            )
        ]
    else:
        texts = [str(pattern_text) for pattern_text in pattern_texts]
    if len(ids) != len(set(ids)):
        raise ValueError("pattern_id가 중복됩니다.")
    if any(not str(text).strip() for text in texts):
        raise ValueError("빈 pattern_text가 있습니다.")
    versions = set(table.column("dataset_version").to_pylist())
    if len(versions) > 1:
        raise ValueError(f"dataset_version이 입력 전체에서 일치하지 않습니다: {versions}")
    dataset_version = dataset_version_override or str(next(iter(versions), ""))
    if not dataset_version.strip():
        raise ValueError("dataset_version이 비어 있습니다.")
    if dry_run:
        report = {
            "status": "dry_run",
            "rows": table.num_rows,
            "request_count": (table.num_rows + batch_size - 1) // batch_size,
            "model": model,
            "dimensions": dimensions,
            "dataset_version": dataset_version,
            "input_sha256": _sha256(input_path),
            "embedding_text_mode": embedding_text_mode,
            "embedding_text_version": embedding_text_version,
        }
        output_path.with_suffix(".report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
    checkpoints = checkpoint_dir or output_path.parent / f"{output_path.stem}_checkpoints"
    input_sha256 = _sha256(input_path)
    if embedder is None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        from langchain_openai import OpenAIEmbeddings

        embedder = OpenAIEmbeddings(
            model=model,
            api_key=settings.openai_api_key,
            dimensions=dimensions,
        ).embed_documents
    vectors: list[list[float]] = []
    checkpoints.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoints / "manifest.json"
    manifest = {
        "input_sha256": input_sha256,
        "pattern_ids": ids,
        "model": model,
        "dimensions": dimensions,
        "batch_size": batch_size,
        "embedding_text_mode": embedding_text_mode,
        "embedding_text_version": embedding_text_version,
        "dataset_version": dataset_version,
    }
    if manifest_path.exists() and resume:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError("기존 패턴 임베딩 checkpoint 설정이 현재 입력과 다릅니다.")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        end = min(start + batch_size, len(texts))
        checkpoint = checkpoints / f"batch_{batch_number:06d}.json"
        batch_ids = ids[start:end]
        batch_vectors: list[list[float]] | None = None
        if resume and checkpoint.exists():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if cached.get("pattern_ids") == batch_ids:
                batch_vectors = cached.get("vectors")
        if batch_vectors is None:
            batch_vectors = embedder(texts[start:end])
            checkpoint.write_text(
                json.dumps({"pattern_ids": batch_ids, "vectors": batch_vectors}),
                encoding="utf-8",
            )
        vectors.extend(batch_vectors)
    if len(vectors) != len(texts) or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("패턴 임베딩 개수 또는 차원이 일치하지 않습니다.")
    columns = {name: table.column(name) for name in table.column_names}
    columns["embedding"] = pa.array(vectors, type=pa.list_(pa.float32(), list_size=dimensions))
    columns["embedding_model"] = pa.array([model] * len(texts))
    columns["embedding_dimensions"] = pa.array([dimensions] * len(texts), type=pa.int32())
    columns["embedding_text_mode"] = pa.array([embedding_text_mode] * len(texts))
    columns["embedding_text_version"] = pa.array([embedding_text_version] * len(texts))
    columns["dataset_version"] = pa.array([dataset_version] * len(texts))
    result = pa.table(columns)
    result = result.replace_schema_metadata(
        {
            b"dataset_version": dataset_version.encode("utf-8"),
            b"embedding_model": model.encode("utf-8"),
            b"embedding_dimensions": str(dimensions).encode("utf-8"),
            b"embedding_text_mode": embedding_text_mode.encode("utf-8"),
            b"embedding_text_version": embedding_text_version.encode("utf-8"),
            b"input_sha256": input_sha256.encode("ascii"),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(result, output_path, compression="zstd")
    report = {"status": "success", "rows": len(texts), "model": model, "dimensions": dimensions, "embedding_text_mode": embedding_text_mode, "embedding_text_version": embedding_text_version, "output": str(output_path), "checkpoint_dir": str(checkpoints)}
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = _args()
    if not args.dry_run and not args.confirm_run:
        raise SystemExit("실제 임베딩 API 호출에는 --confirm-run이 필요합니다.")
    print(json.dumps(embed_patterns(args.input, args.output, batch_size=args.batch_size, limit=args.limit, dry_run=args.dry_run, model=args.model, dimensions=args.dimensions, checkpoint_dir=args.checkpoint_dir, embedding_text_mode=args.embedding_text_mode, embedding_text_version=args.embedding_text_version, dataset_version_override=args.dataset_version), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
