"""Private GitHub Release에 첨부할 정적 데이터 manifest를 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parquet_metadata(path: Path) -> dict[str, str | int]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[])
    metadata = {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in (table.schema.metadata or {}).items()
    }
    required = {
        "embedding_model",
        "embedding_dimensions",
        "embedding_text_mode",
        "embedding_text_version",
        "dataset_version",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Parquet 메타데이터가 부족합니다: {missing}")
    if metadata["embedding_dimensions"].isdigit():
        metadata["embedding_dimensions"] = int(metadata["embedding_dimensions"])
    return metadata


def create_manifest(
    artifact: Path,
    output: Path,
    *,
    release_tag: str,
    asset_name: str | None = None,
    dataset_version: str,
) -> dict[str, str | int]:
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    parquet_metadata = _parquet_metadata(artifact)
    artifact_dataset_version = str(parquet_metadata["dataset_version"])
    if artifact_dataset_version != dataset_version:
        raise ValueError(
            f"dataset_version이 Parquet과 다릅니다: parquet={artifact_dataset_version}, requested={dataset_version}"
        )
    manifest: dict[str, str | int] = {
        "dataset_version": dataset_version,
        "release_tag": release_tag,
        "asset_name": asset_name or artifact.name,
        "format": "parquet",
        "rows": _row_count(artifact),
        "sha256": _sha256(artifact),
        "embedding_model": parquet_metadata["embedding_model"],
        "embedding_dimensions": parquet_metadata["embedding_dimensions"],
        "embedding_text_mode": parquet_metadata["embedding_text_mode"],
        "embedding_text_version": parquet_metadata["embedding_text_version"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def validate_manifest(artifact: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = _sha256(artifact)
    if actual != manifest.get("sha256"):
        raise ValueError("Parquet checksum이 manifest와 다릅니다.")
    rows = _row_count(artifact)
    if rows != manifest.get("rows"):
        raise ValueError("Parquet 행 수가 manifest와 다릅니다.")
    parquet_metadata = _parquet_metadata(artifact)
    for key in (
        "dataset_version",
        "embedding_model",
        "embedding_dimensions",
        "embedding_text_mode",
        "embedding_text_version",
    ):
        if str(parquet_metadata[key]) != str(manifest.get(key)):
            raise ValueError(f"Parquet 메타데이터가 manifest와 다릅니다: {key}")
    return {"status": "valid", "dataset_version": manifest["dataset_version"], "rows": rows, "sha256": actual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--artifact", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--release-tag", required=True)
    create.add_argument("--dataset-version", required=True)
    create.add_argument("--asset-name")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        result = create_manifest(args.artifact, args.output, release_tag=args.release_tag, asset_name=args.asset_name, dataset_version=args.dataset_version)
    else:
        result = validate_manifest(args.artifact, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
