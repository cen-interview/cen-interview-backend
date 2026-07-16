import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_MODULE_PATH = Path(__file__).parents[2] / "scripts" / "question_dataset" / "create_dataset_manifest.py"
_SPEC = importlib.util.spec_from_file_location("create_dataset_manifest", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
create_manifest = _MODULE.create_manifest
validate_manifest = _MODULE.validate_manifest


def test_manifest_preserves_embedding_metadata(tmp_path) -> None:
    artifact = tmp_path / "patterns.parquet"
    manifest = tmp_path / "manifest.json"
    table = pa.table({"pattern_id": ["p1"], "embedding": [[[0.1, 0.2]]]})
    table = table.replace_schema_metadata(
        {
            b"dataset_version": b"question-patterns-v1-candidate-2",
            b"embedding_model": b"text-embedding-3-small",
            b"embedding_dimensions": b"1536",
            b"embedding_text_mode": b"pattern_with_signals",
            b"embedding_text_version": b"1",
        }
    )
    pq.write_table(table, artifact)

    result = create_manifest(
        artifact,
        manifest,
        release_tag="question-patterns-v1-candidate-2",
        dataset_version="question-patterns-v1-candidate-2",
    )

    assert result["embedding_text_mode"] == "pattern_with_signals"
    assert validate_manifest(artifact, manifest)["status"] == "valid"
