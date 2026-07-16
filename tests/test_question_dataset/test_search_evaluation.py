import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "question_dataset"
    / "evaluate_question_pattern_search.py"
)
_SPEC = importlib.util.spec_from_file_location("question_pattern_search_evaluation", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

from question_pattern_search_evaluation import (  # noqa: E402
    _classify_failure,
    _metric,
    evaluation_queries,
)


def test_evaluation_queries_include_relevant_irrelevant_and_boundary_groups() -> None:
    queries = evaluation_queries()

    assert len(queries) == 51
    assert sum(query.relevant for query in queries) == 35
    assert sum(query.query_group == "irrelevant" for query in queries) == 8
    assert sum(query.query_group == "boundary" for query in queries) == 8
    assert all("pattern_text" not in query.text for query in queries)


def test_failure_classification_distinguishes_threshold_and_taxonomy() -> None:
    query = evaluation_queries()[0]
    assert _classify_failure(query, [], 0.55) == "threshold_problem"
    assert _classify_failure(query, [], None) == "actual_pattern_shortage"
    result = {
        "signal_kind": "technical_pattern",
        "topic_family": "architecture_design",
        "similarity": 0.70,
        "embedding": [1.0, 0.0],
    }
    assert _classify_failure(query, [result], None) == "topic_taxonomy_mismatch"


def test_metric_separates_no_result_and_wrong_result() -> None:
    queries = evaluation_queries()[:2]
    retrievals = [
        {"query_id": queries[0].query_id, "results": []},
        {
            "query_id": queries[1].query_id,
            "results": [
                {
                    "pattern_id": "wrong",
                    "pattern_text": "무관한 결과",
                    "signal_kind": "evidence_grounded_frame",
                    "topic_family": "collaboration",
                    "similarity": 0.70,
                    "embedding": [1.0, 0.0],
                }
            ],
        },
    ]

    metric = _metric(retrievals, queries, 0.55)

    assert metric["no_result_count"] == 1
    assert metric["wrong_result_count"] == 1
