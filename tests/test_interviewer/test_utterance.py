"""Interviewer 5단계 발화 조립과 LLM 폴백 단위 테스트."""

from types import SimpleNamespace

import pytest

from interview.interviewer.session import SessionState
from interview.interviewer.speech.composition import compose_utterance
from interview.interviewer.speech import utterance
from interview.interviewer.workflow.runtime import InterviewDeps
from interview.schemas.question import Difficulty, Question, QuestionCategory, QuestionKind


def make_question(kind: QuestionKind = QuestionKind.MAIN) -> Question:
    """발화 테스트에 사용할 고정 질문을 만든다.

    Args:
        kind:
            생성할 질문의 종류.

    Returns:
        원문 변경 여부를 확인할 수 있는 테스트용 Question.
    """
    return Question(
        question_id=f"q-{kind.value}",
        text="FastAPI의 의존성 주입 방식에 대해 설명해 주세요.",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=kind,
        category=QuestionCategory.TECHNICAL,
    )


class FailIfCalledLlm:
    """발화 조립이 LLM에 접근하면 즉시 실패하는 테스트 대역."""

    def with_structured_output(self, _schema: type):
        raise AssertionError("template utterance must not call an LLM")


def make_runtime(llm: object | None = None) -> SimpleNamespace:
    """compose_utterance에 전달할 LangGraph runtime 대역을 만든다.

    Args:
        llm:
            발화 생성에 사용할 선택적 LLM fake.

    Returns:
        InterviewDeps를 context로 가진 간단한 runtime 대역.
    """
    deps = InterviewDeps(strategy=object(), assessment=object(), llm=llm)
    return SimpleNamespace(context=deps)


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (utterance.greeting, "안녕하세요. 지금부터 면접을 시작하겠습니다."),
        (utterance.question, "네, 답변 잘 들었습니다."),
        (utterance.follow_up, "네, 말씀 감사합니다."),
        (utterance.challenge, "네, 답변 감사합니다."),
        (utterance.hint, "답변이 어려우시다면 다음 관점에서 생각해 보셔도 좋습니다."),
        (utterance.replay, "네, 질문을 다시 말씀드리겠습니다."),
        (utterance.pause_prompt, "잠시 쉬었다가 면접을 계속 진행하시겠습니까?"),
        (utterance.closing, "이상으로 면접을 마치겠습니다. 참여해 주셔서 감사합니다."),
    ],
)
def test_template_returns_expected_text(template, expected):
    """상황별 템플릿은 외부 의존성 없이 항상 고정 문장을 반환한다.

    Args:
        template:
            호출할 상황별 템플릿 함수.

        expected:
            템플릿이 반환해야 하는 고정 문장.
    """
    assert template() == expected


@pytest.mark.parametrize(
    ("kind", "expected_preamble"),
    [
        (QuestionKind.MAIN, utterance.question()),
        (QuestionKind.FOLLOW_UP, utterance.follow_up()),
        (QuestionKind.CHALLENGE, utterance.challenge()),
        (QuestionKind.CONFIRM_POSITIVE, utterance.follow_up()),
        (QuestionKind.CONFIRM_NEGATIVE, utterance.challenge()),
        (QuestionKind.TRAP, utterance.question()),
        (QuestionKind.HINT, utterance.hint()),
    ],
)
def test_compose_utterance_uses_template_for_question_kind(kind, expected_preamble):
    """LLM이 없으면 질문 종류에 맞는 템플릿과 원본 질문을 조립한다.

    Args:
        kind:
            발화 조립에 사용할 질문 종류.

        expected_preamble:
            질문 종류에 대응해야 하는 기본 안내 문장.
    """
    question = make_question(kind)
    original_text = question.text
    state = SessionState(
        session_id="session-template",
        current_question=question,
        turn_type="question",
    )

    result = compose_utterance(state, make_runtime())

    assert question.text == original_text
    assert result["last_utterance"] == f"{expected_preamble}\n\n{original_text}"
    assert result["utterance_queue"] == [expected_preamble, original_text]
    assert result["transcript"][-1].role == "interviewer"
    assert result["transcript"][-1].question_id == question.question_id
    assert result["transcript"][-1].kind == kind.value


def test_replay_situation_has_priority_over_question_kind():
    """replay 상황에서는 질문 종류와 무관하게 재제시 안내를 사용한다."""
    question = make_question(QuestionKind.CHALLENGE)
    state = SessionState(
        session_id="session-replay",
        current_question=question,
        turn_type="replay",
    )

    result = compose_utterance(state, make_runtime())

    assert result["last_utterance"] == f"{utterance.replay()}\n\n{question.text}"


def test_closing_does_not_append_stale_current_question():
    """종료 발화에는 상태에 남아 있는 이전 질문 본문을 덧붙이지 않는다."""
    question = make_question()
    state = SessionState(
        session_id="session-closing",
        current_question=question,
        turn_type="closing",
        finished=True,
    )

    result = compose_utterance(state, make_runtime())

    assert result["last_utterance"] == utterance.closing()
    assert question.text not in result["last_utterance"]
    assert result["transcript"][-1].question_id is None
    assert result["transcript"][-1].kind is None


def test_compose_utterance_never_calls_injected_llm():
    """LLM이 주입돼 있어도 발화는 평가별 템플릿으로만 조립한다."""
    question = make_question()
    state = SessionState(
        session_id="session-template-only",
        current_question=question,
        turn_type="question",
        last_signal={"quality": "sufficient"},
    )

    result = compose_utterance(state, make_runtime(FailIfCalledLlm()))

    assert result["last_utterance"] == f"{utterance.sufficient()}\n\n{question.text}"
    assert result["error"] is None
