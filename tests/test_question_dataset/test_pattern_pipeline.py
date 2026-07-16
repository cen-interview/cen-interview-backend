from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.question_dataset.build_interview_question_patterns import build_patterns
from scripts.question_dataset.collect_cluster_pattern_batch import collect_cluster_batch
from scripts.question_dataset.embed_question_patterns import embed_patterns


def _write_tagged(path: Path) -> None:
    table = pa.table(
        {
            "cluster_id": ["c1", "c2", "c3"],
            "cluster_frequency": [10, 3, 1],
            "is_interview_question": [True, True, False],
            "signal_kind": ["technical_pattern", "evidence_grounded_frame", "excluded"],
            "pattern_text": ["HTTP와 HTTPS의 차이는?", "프로젝트에서 가장 어려웠던 문제는?", ""],
            "required_evidence_signals": [[], ["problem", "resolution"], []],
            "topic_family": ["network_web_api", "problem_solving", "other"],
        }
    ).replace_schema_metadata({b"dataset_version": b"dataset-v1"})
    pq.write_table(table, path)


def test_build_patterns_removes_excluded_and_keeps_signal_fields(tmp_path: Path) -> None:
    tagged = tmp_path / "tagged.parquet"
    output = tmp_path / "patterns.parquet"
    _write_tagged(tagged)

    report = build_patterns(tagged, output)

    rows = pq.read_table(output).to_pylist()
    assert report["output_rows"] == 2
    assert [row["pattern_id"] for row in rows] == ["c1", "c2"]
    assert rows[1]["required_evidence_signals"] == ["problem", "resolution"]


def test_pattern_embedding_dry_run_does_not_call_embedder(tmp_path: Path) -> None:
    tagged = tmp_path / "tagged.parquet"
    patterns = tmp_path / "patterns.parquet"
    embedded = tmp_path / "embedded.parquet"
    _write_tagged(tagged)
    build_patterns(tagged, patterns)

    report = embed_patterns(
        patterns,
        embedded,
        dimensions=3,
        dry_run=True,
        embedder=lambda _texts: pytest.fail("dry-run에서 임베딩을 호출하면 안 됩니다."),
    )

    assert report["status"] == "dry_run"
    assert report["rows"] == 2
    assert not embedded.exists()


def test_pattern_embedding_writes_configured_dimension(tmp_path: Path) -> None:
    tagged = tmp_path / "tagged.parquet"
    patterns = tmp_path / "patterns.parquet"
    embedded = tmp_path / "embedded.parquet"
    _write_tagged(tagged)
    build_patterns(tagged, patterns)

    report = embed_patterns(
        patterns,
        embedded,
        dimensions=3,
        embedder=lambda texts: [[float(index), 0.0, 1.0] for index, _ in enumerate(texts)],
    )

    table = pq.read_table(embedded)
    assert report["status"] == "success"
    assert table.schema.field("embedding").type.list_size == 3
    assert json.loads((embedded.with_suffix(".report.json")).read_text())[
        "rows"
    ] == 2


def test_collect_cluster_batch_validates_and_joins_minimal_output(tmp_path: Path) -> None:
    clusters = tmp_path / "clusters.parquet"
    tagged = tmp_path / "tagged.parquet"
    manifest = tmp_path / "manifest.json"
    job = tmp_path / "job.json"
    report = tmp_path / "report.json"
    pq.write_table(
        pa.table(
            {
                "cluster_id": ["c1"],
                "representative_text": ["HTTP란?"],
                "member_questions": [["HTTP란?"]],
                "cluster_frequency": [2],
            }
        ),
        clusters,
    )
    manifest.write_text(
        json.dumps({"selected_cluster_ids": ["c1"], "dataset_version": "dataset-v1"}),
        encoding="utf-8",
    )
    job.write_text(
        json.dumps({"status": "in_progress", "batch_id": "batch-1"}), encoding="utf-8"
    )
    response = {
        "response": {
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "cluster_id": "c1",
                                            "is_interview_question": True,
                                            "signal_kind": "technical_pattern",
                                            "pattern_text": "HTTP와 HTTPS의 차이는?",
                                            "required_evidence_signals": [],
                                            "topic_family": "network_web_api",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
        }
    }

    class FakeContent:
        text = json.dumps(response)

    class FakeFiles:
        @staticmethod
        def content(_file_id: str) -> FakeContent:
            return FakeContent()

    class FakeBatches:
        @staticmethod
        def retrieve(_batch_id: str) -> object:
            return type("RemoteBatch", (), {"status": "completed", "output_file_id": "file-1"})()

    class FakeClient:
        files = FakeFiles()
        batches = FakeBatches()

    result = collect_cluster_batch(manifest, job, clusters, tagged, report, client=FakeClient())

    assert result["status"] == "success"
    tagged_table = pq.read_table(tagged)
    assert tagged_table.to_pylist()[0]["pattern_text"] == "HTTP와 HTTPS의 차이는?"
    assert tagged_table.schema.metadata[b"dataset_version"] == b"dataset-v1"
    assert json.loads(job.read_text(encoding="utf-8"))["status"] == "completed"
