"""계약(schemas) 동작 검증. 이미 구현돼 있으니 바로 통과해야 한다."""

from interview.schemas.report import CompetencyModel

# 주의: CompetencyModel 은 메서드 없는 순수 데이터 모델(BaseModel)이다.
# record()/strengths()/weaknesses() 같은 메서드는 없다 — 누적/판정 로직은
# 호출부(assessment/agent.py)가 topic_scores 등 필드를 직접 갱신해서 한다.


def test_competency_model_fields():
    c = CompetencyModel(
        topic_scores={"JPA": 0.85, "Spring": 0.3},
        strengths=["JPA"],
        weaknesses=["Spring"],
    )
    assert "JPA" in c.strengths
    assert "Spring" in c.weaknesses
    assert c.topic_scores["JPA"] == 0.85


def test_signal_fixture(shallow_signal):
    assert shallow_signal.missing_keywords  # 비어있지 않음
