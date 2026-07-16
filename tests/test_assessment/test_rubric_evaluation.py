from interview.assessment.agent import AssessmentAgent
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)
from interview.schemas.rubric import RubricMatchResult


def _technical_question() -> Question:
    return Question(
        question_id="new-session-question",
        text="FastAPI의 비동기 함수는 언제 사용하나요?",
        topic="FastAPI",
        difficulty=Difficulty.MEDIUM,
        kind=QuestionKind.MAIN,
        category=QuestionCategory.TECHNICAL,
    )


def test_matching_rubric_skips_llm_assessment(monkeypatch):
    class MatchingStore:
        def match(self, **_):
            return RubricMatchResult(
                question_id="new-session-question",
                rubric_version="v1",
                required_criteria_count=2,
                matched_required_count=2,
                matched_rubric_question_id="previous-session-question",
                question_similarity=0.93,
            )

    monkeypatch.setattr(
        "interview.assessment.agent.get_rubric_store",
        lambda: MatchingStore(),
    )
    monkeypatch.setattr(
        "interview.assessment.agent.get_compiled_graph",
        lambda: (_ for _ in ()).throw(AssertionError("LLM graph was called")),
    )

    signal = AssessmentAgent().evaluate(
        _technical_question(),
        "비동기 I/O 작업에서는 async def를 사용합니다.",
    )

    assert signal.evaluation_source == "rubric"
    assert signal.rubric_version == "v1"
    assert signal.rubric_question_similarity == 0.93
    assert signal.accuracy == 1.0
    assert signal.sufficiency == 1.0


def test_incomplete_rubric_match_falls_back_to_llm(monkeypatch):
    class IncompleteStore:
        def match(self, **_):
            return RubricMatchResult(
                question_id="new-session-question",
                rubric_version="v1",
                required_criteria_count=2,
                matched_required_count=1,
            )

    class LlmGraph:
        def invoke(self, state):
            raise RuntimeError("llm fallback reached")

    monkeypatch.setattr(
        "interview.assessment.agent.get_rubric_store",
        lambda: IncompleteStore(),
    )
    monkeypatch.setattr(
        "interview.assessment.agent.get_compiled_graph",
        lambda: LlmGraph(),
    )

    try:
        AssessmentAgent().evaluate(_technical_question(), "불완전한 답변")
    except RuntimeError as exc:
        assert str(exc) == "llm fallback reached"
    else:
        raise AssertionError("incomplete rubric match did not use LLM fallback")


def test_three_of_six_required_criteria_are_sufficient():
    match = RubricMatchResult(
        question_id="new-session-question",
        rubric_version="v1",
        required_criteria_count=6,
        matched_required_count=3,
    )

    assert match.required_coverage == 3 / 6
    assert match.is_sufficient is True
    assert match.all_required_matched is False


def test_two_of_six_required_criteria_fall_back_to_llm():
    match = RubricMatchResult(
        question_id="new-session-question",
        rubric_version="v1",
        required_criteria_count=6,
        matched_required_count=2,
    )

    assert match.is_sufficient is False
