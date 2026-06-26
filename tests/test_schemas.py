"""계약(schemas) 동작 검증. 이미 구현돼 있으니 바로 통과해야 한다."""

from interview.schemas.report import CompetencyModel


def test_competency_strengths_weaknesses():
    c = CompetencyModel()
    c.record("JPA", 0.9)
    c.record("JPA", 0.8)
    c.record("Spring", 0.3)
    assert "JPA" in c.strengths()
    assert "Spring" in c.weaknesses()


def test_signal_fixture(shallow_signal):
    assert shallow_signal.missing_keywords  # 비어있지 않음
