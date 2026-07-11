"""질문 생성 그래프(strategy/graph.py)의 노드/엣지 단위 테스트."""

from interview.schemas.evidence import CoverageMap, TopicCoverage
from interview.strategy.graph import QuestionGenState, pick_topic, route_after_retrieve


def test_route_after_retrieve_goes_to_pick_topic_when_no_evidence_and_retries_left():
    state = QuestionGenState(evidence_chunks=[], retry_count=0)
    assert route_after_retrieve(state) == "pick_topic"


def test_route_after_retrieve_gives_up_when_retry_limit_reached():
    state = QuestionGenState(evidence_chunks=[], retry_count=3)
    assert route_after_retrieve(state) == "generate"


def test_pick_topic_avoids_previously_tried_topic_on_retry():
    coverage = CoverageMap(
        topic_coverage={
            "FastAPI": TopicCoverage(confidence=0.9, chunk_count=5),
            "Docker": TopicCoverage(confidence=0.8, chunk_count=5),
        }
    )
    state = QuestionGenState(coverage=coverage, topic="FastAPI", retry_count=0)

    updates = pick_topic(state)

    assert updates["topic"] != "FastAPI"
    assert "FastAPI" in updates["tried_topics"]
    assert updates["retry_count"] == 1