from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.question_dataset.embed_question_candidates import embed_candidates


pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq


def _write_input(path: Path, ids: list[str] | None = None) -> None:
    question_ids = ids or ["q1", "q2", "q3"]
    pq.write_table(
        pa.table(
            {
                "item_id": question_ids,
                "text": ["HTTP와 HTTPS 차이는?", "Redis를 선택한 이유는?", "트랜잭션이란?"][: len(question_ids)],
                "frequency": [2, 3, 1][: len(question_ids)],
            }
        ),
        path,
    )


def test_dry_run_reports_volume_without_calling_embedder(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _write_input(input_path)

    def fail_embedder(_: list[str]) -> list[list[float]]:
        raise AssertionError("dry-run에서 임베딩을 호출하면 안 됩니다.")

    report = embed_candidates(
        input_path,
        tmp_path / "output.parquet",
        tmp_path / "report.json",
        dry_run=True,
        batch_size=2,
        concurrency=1,
        model="text-embedding-3-small",
        dimensions=4,
        embedder=fail_embedder,
    )

    assert report["status"] == "dry_run"
    assert report["input"]["rows"] == 3
    assert report["embedding"]["request_count"] == 2
    assert not (tmp_path / "output.parquet").exists()


def test_embedding_is_checkpointed_and_final_output_is_reused(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    report_path = tmp_path / "report.json"
    checkpoint_dir = tmp_path / "checkpoints"
    _write_input(input_path)
    calls: list[list[str]] = []

    def fake_embedder(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[float(index), 0.1, 0.2, 0.3] for index, _ in enumerate(texts)]

    first = embed_candidates(
        input_path,
        output_path,
        report_path,
        checkpoint_dir=checkpoint_dir,
        batch_size=2,
        concurrency=1,
        model="test-embedding",
        dimensions=4,
        embedder=fake_embedder,
    )

    assert first["status"] == "success"
    assert len(calls) == 2
    table = pq.read_table(output_path)
    assert table.column_names == [
        "question_id",
        "normalized_text",
        "exact_frequency",
        "embedding",
        "embedding_model",
        "embedding_dimensions",
        "dataset_version",
    ]
    assert table.column("question_id").to_pylist() == ["q1", "q2", "q3"]

    def should_not_run(_: list[str]) -> list[list[float]]:
        raise AssertionError("완료 결과 재사용 시 API를 다시 호출하면 안 됩니다.")

    second = embed_candidates(
        input_path,
        output_path,
        report_path,
        checkpoint_dir=checkpoint_dir,
        batch_size=2,
        concurrency=1,
        model="test-embedding",
        dimensions=4,
        embedder=should_not_run,
    )

    assert second["status"] == "success"
    assert second["reused_batches"] == 2


def test_duplicate_question_ids_are_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _write_input(input_path, ids=["same", "same"])

    with pytest.raises(ValueError, match="중복"):
        embed_candidates(
            input_path,
            tmp_path / "output.parquet",
            tmp_path / "report.json",
            dry_run=True,
            dimensions=4,
        )
