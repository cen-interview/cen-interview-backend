from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.question_dataset.prepare_cluster_pattern_batch import prepare_cluster_batch


def test_prepares_minimal_cluster_batch_without_submission(tmp_path: Path) -> None:
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    input_path = tmp_path / "clusters.parquet"
    output_path = tmp_path / "batch.jsonl"
    manifest_path = tmp_path / "manifest.json"
    pq.write_table(
        pa.table(
            {
                "cluster_id": ["c1", "c2"],
                "representative_text": ["HTTP란?", "프로젝트에서 어려웠던 점은?"],
                "member_questions": [["HTTP란?"], ["어려웠던 점은?", "해결한 문제는?"]],
                "cluster_frequency": [3, 2],
            }
        ),
        input_path,
    )

    manifest = prepare_cluster_batch(
        input_path,
        output_path,
        manifest_path,
        model="test-model",
        items_per_request=2,
        use_all=True,
    )

    line = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    properties = line["body"]["response_format"]["json_schema"]["schema"]["properties"]
    assert manifest["status"] == "prepared"
    assert manifest["selected_clusters"] == 2
    assert manifest["request_count"] == 1
    assert set(properties["items"]["items"]["required"]) == {
        "cluster_id",
        "is_interview_question",
        "signal_kind",
        "pattern_text",
        "required_evidence_signals",
        "topic_family",
    }
