from interview.assessment.agent import AssessmentAgent
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)
from interview.schemas.rubric import RubricMatchResult
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


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


def test_rubric_source_selection_uses_embedding_store_without_generation_llm(
    monkeypatch,
):
    filtered_sources = []

    class NoMatchStore:
        def match(self, **_):
            return None

        def filter_novel_questions(self, sources):
            filtered_sources.extend(sources)
            return sources

    class SufficientGraph:
        def invoke(self, state):
            return {
                "evidence_chunks": [],
                "final_signal": AnswerQualitySignal(
                    answer_id="answer-llm",
                    question_id=state.question.question_id,
                    quality=AnswerQuality.SUFFICIENT,
                    accuracy=1.0,
                    sufficiency=1.0,
                ),
            }

    monkeypatch.setattr(
        "interview.assessment.agent.get_rubric_store",
        lambda: NoMatchStore(),
    )
    monkeypatch.setattr(
        "interview.assessment.agent.get_compiled_graph",
        lambda: SufficientGraph(),
    )
    agent = AssessmentAgent()
    question = _technical_question()
    answer = "네트워크 같은 비동기 I/O 작업에서 async def를 사용합니다."
    agent.evaluate(question, answer)
    agent.complete_question_set(question.question_id)

    assert agent.rubric_candidates == []

    sources = agent.collect_rubric_sources()

    assert len(sources) == 1
    assert filtered_sources[0].question_id == question.question_id
    assert filtered_sources[0].answer == answer
    assert agent.rubric_candidates == []
