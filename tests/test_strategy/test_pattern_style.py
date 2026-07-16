"""사후 화법 폴리시(apply_pattern_style, graph.polish_style) 단위 테스트.

실제 LLM/pgvector 호출 없이, question_pattern_store의 계약(InterviewQuestionSignal)에
맞는 가짜 결과로 제어 흐름만 검증한다. (tests/conftest.py의 협업 패턴과 동일)
"""

from interview.schemas.question_pattern import InterviewQuestionSignal
from interview.strategy import graph as graph_module
from interview.strategy import question_gen
from interview.strategy.graph import QuestionGenState, polish_style
from interview.strategy.question_gen import StyledQuestion, apply_pattern_style


def _signal(pattern_id: str = "p1", similarity: float = 0.8) -> InterviewQuestionSignal:
    return InterviewQuestionSignal(
        pattern_id=pattern_id,
        pattern_text="이 부분을 실제로 어떻게 구현하셨는지 설명해 주시겠어요?",
        frequency=10,
        signal_kind="technical_pattern",
        required_evidence_signals=[],
        topic_family="language_framework",
        similarity=similarity,
    )


def test_apply_pattern_style_returns_original_when_no_patterns_matched(monkeypatch):
    monkeypatch.setattr(question_gen, "search_interview_question_signals", lambda **kwargs: [])

    result = apply_pattern_style("FastAPI의 의존성 주입은 어떻게 동작하나요?")

    assert result == "FastAPI의 의존성 주입은 어떻게 동작하나요?"


def test_apply_pattern_style_returns_original_when_search_fails(monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(question_gen, "search_interview_question_signals", _raise)

    result = apply_pattern_style("FastAPI의 의존성 주입은 어떻게 동작하나요?")

    assert result == "FastAPI의 의존성 주입은 어떻게 동작하나요?"


def test_apply_pattern_style_returns_original_on_blank_input(monkeypatch):
    called = False

    def _search(**kwargs):
        nonlocal called
        called = True
        return [_signal()]

    monkeypatch.setattr(question_gen, "search_interview_question_signals", _search)

    assert apply_pattern_style("   ") == "   "
    assert called is False


class _FakeStructuredLLM:
    def __init__(self, styled_text: str) -> None:
        self._styled_text = styled_text

    def invoke(self, messages):
        return StyledQuestion(text=self._styled_text)


class _FakeLLM:
    def __init__(self, styled_text: str) -> None:
        self._styled_text = styled_text

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self._styled_text)


def test_apply_pattern_style_uses_llm_result_when_pattern_matched(monkeypatch):
    monkeypatch.setattr(
        question_gen, "search_interview_question_signals", lambda **kwargs: [_signal()]
    )
    monkeypatch.setattr(question_gen, "get_llm", lambda temperature=0.4: _FakeLLM("다듬어진 질문인가요?"))

    result = apply_pattern_style("원본 질문인가요?")

    assert result == "다듬어진 질문인가요?"


def test_apply_pattern_style_falls_back_to_original_when_llm_fails(monkeypatch):
    class _FailingStructuredLLM:
        def invoke(self, messages):
            raise RuntimeError("llm down")

    class _FailingLLM:
        def with_structured_output(self, schema):
            return _FailingStructuredLLM()

    monkeypatch.setattr(
        question_gen, "search_interview_question_signals", lambda **kwargs: [_signal()]
    )
    monkeypatch.setattr(question_gen, "get_llm", lambda temperature=0.4: _FailingLLM())

    result = apply_pattern_style("원본 질문인가요?")

    assert result == "원본 질문인가요?"


def test_graph_polish_style_node_updates_generated_text(monkeypatch):
    monkeypatch.setattr(graph_module, "apply_pattern_style", lambda text: f"styled:{text}")

    state = QuestionGenState(generated_text="원본 질문")
    updates = polish_style(state)

    assert updates == {"generated_text": "styled:원본 질문"}


def test_get_compiled_graph_wires_polish_style_between_validate_and_build_result():
    """새 노드가 붙어도 그래프가 정상적으로 컴파일되는지(엣지 오타 등) 확인."""
    compiled = graph_module.get_compiled_graph()

    node_names = set(compiled.get_graph().nodes.keys())
    assert "polish_style" in node_names
