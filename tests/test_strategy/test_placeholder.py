"""Strategy fake 구현 테스트."""

from interview.schemas.evidence import CoverageMap
from interview.strategy import StrategyAgent


def test_strategy_generates_first_question_from_coverage():
    strategy = StrategyAgent(
        CoverageMap(topic_confidence={"JPA": 0.8, "JWT": 0.6})
    )

    question = strategy.next_question(last_signal=None)

    assert question.kind == "main"
    assert question.topic == "JPA"
    assert strategy.state.question_count == 1


def test_strategy_generates_follow_up_with_missing_keywords():
    strategy = StrategyAgent(CoverageMap(topic_confidence={"JPA": 0.8}))

    question = strategy.next_follow_up("JPA", ["fetch join"])

    assert question.kind == "follow_up"
    assert question.topic == "JPA"
    assert "fetch join" in question.text
