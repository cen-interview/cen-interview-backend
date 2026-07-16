"""CSV 질문 패턴 적재 로직을 검증한다."""

import csv
import json
from pathlib import Path

import pytest

from interview.evidence import question_pattern_ingestion


def _write_csv(path: Path, *, dimensions: int = 3) -> None:
    fieldnames = [
        "pattern_id",
        "pattern_text",
        "frequency",
        "signal_kind",
        "required_evidence_signals",
        "topic_family",
        "embedding",
        "dataset_version",
        "variants",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "pattern_id": "pattern-1",
                "pattern_text": "기술 선택 이유를 설명해 주세요.",
                "frequency": "3",
                "signal_kind": "experience_frame",
                "required_evidence_signals": json.dumps(["technical_choice"]),
                "topic_family": "technical_choice",
                "embedding": json.dumps([0.1] * dimensions),
                "dataset_version": "question-patterns-v1",
                "variants": json.dumps(["왜 이 기술을 선택했나요?"]),
            }
        )


def test_ingest_question_patterns_replaces_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "interview_question_patterns_test.csv"
    _write_csv(csv_path)
    monkeypatch.setattr(question_pattern_ingestion.settings, "embedding_dimensions", 3)
    calls: dict[str, object] = {}

    class FakeStore:
        def replace_dataset(self, rows: list[dict[str, object]], dataset_version: str) -> int:
            calls["rows"] = rows
            calls["dataset_version"] = dataset_version
            return len(rows)

    monkeypatch.setattr(
        question_pattern_ingestion, "get_question_pattern_store", lambda: FakeStore()
    )

    result = question_pattern_ingestion.ingest_question_patterns(csv_path)

    assert result == 1
    assert calls["dataset_version"] == "question-patterns-v1"
    rows = calls["rows"]
    assert isinstance(rows, list)
    assert rows[0]["embedding"] == [0.1, 0.1, 0.1]


def test_load_question_pattern_rows_rejects_wrong_embedding_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "interview_question_patterns_test.csv"
    _write_csv(csv_path, dimensions=2)
    monkeypatch.setattr(question_pattern_ingestion.settings, "embedding_dimensions", 3)

    with pytest.raises(ValueError, match="embedding 차원"):
        question_pattern_ingestion.load_question_pattern_rows(csv_path)
