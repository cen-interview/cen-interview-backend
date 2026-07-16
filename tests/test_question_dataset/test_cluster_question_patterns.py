from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.question_dataset.cluster_question_patterns import cluster_patterns


pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq


def _write_embeddings(path: Path) -> None:
    table = pa.table(
        {
            "question_id": ["q1", "q2", "q3"],
            "normalized_text": ["HTTP란?", "HTTP를 설명해보세요", "Redis를 왜 썼나요?"],
            "exact_frequency": [2, 3, 5],
            "embedding": pa.array(
                [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
                type=pa.list_(pa.float32(), list_size=2),
            ),
            "embedding_model": ["test"] * 3,
            "embedding_dimensions": [2] * 3,
            "dataset_version": ["test-v1"] * 3,
        }
    )
    pq.write_table(table, path)


def test_clusters_neighbors_and_sums_exact_frequency(tmp_path: Path) -> None:
    input_path = tmp_path / "embedded.parquet"
    output_path = tmp_path / "clusters.parquet"
    _write_embeddings(input_path)

    neighbors = {
        2: [(2, 1.0)],
        1: [(1, 1.0), (0, 0.95)],
        0: [(0, 1.0), (1, 0.95)],
    }
    report = cluster_patterns(
        input_path,
        output_path,
        tmp_path / "report.json",
        neighbor_search=lambda index, _: neighbors[index],
    )

    clusters = pq.read_table(output_path).to_pylist()
    assert report["distribution"]["cluster_count"] == 2
    assert report["distribution"]["singleton_count"] == 1
    assert clusters[0]["representative_question_id"] == "q3"
    assert clusters[0]["cluster_frequency"] == 5
    assert clusters[1]["representative_question_id"] == "q2"
    assert clusters[1]["member_question_ids"] == ["q1", "q2"]
    assert clusters[1]["cluster_frequency"] == 5


def test_dry_run_does_not_build_ann_index(tmp_path: Path) -> None:
    input_path = tmp_path / "embedded.parquet"
    _write_embeddings(input_path)

    report = cluster_patterns(
        input_path,
        tmp_path / "clusters.parquet",
        tmp_path / "report.json",
        dry_run=True,
        neighbor_search=lambda _index, _count: pytest.fail("dry-run에서 검색하면 안 됩니다."),
    )

    assert report["status"] == "dry_run"
    assert report["input"]["rows"] == 3
    assert not (tmp_path / "clusters.parquet").exists()


def test_usearch_cosine_index_clusters_similar_vectors(tmp_path: Path) -> None:
    input_path = tmp_path / "embedded.parquet"
    output_path = tmp_path / "clusters.parquet"
    _write_embeddings(input_path)

    cluster_patterns(
        input_path,
        output_path,
        tmp_path / "report.json",
        similarity_threshold=0.95,
        neighbors=3,
    )

    clusters = pq.read_table(output_path).to_pylist()
    assert len(clusters) == 2
    assert {tuple(cluster["member_question_ids"]) for cluster in clusters} == {
        ("q3",),
        ("q1", "q2"),
    }
