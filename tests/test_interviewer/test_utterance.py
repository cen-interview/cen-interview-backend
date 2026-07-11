"""Interviewer 5단계 발화 조립과 LLM 폴백 단위 테스트."""

import time
from types import SimpleNamespace
from typing import Any

import pytest

import interview.interviewer.graph as interviewer_graph
from interview.interviewer import utterance
from interview.interviewer.graph import InterviewDeps, compose_utterance
from interview.interviewer.models import ComposedUtterance
from interview.interviewer.session import SessionState, Turn
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


class FakeLlm:
    """구조화 출력과 호출 결과를 제어하는 발화 생성 LLM fake."""

    def __init__(
        self,
        output: Any = None,
        error: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        """반환값, 예외, 지연 시간을 초기화한다.

        Args:
            output:
                invoke가 반환할 구조화 출력 또는 dict.

            error:
                invoke에서 발생시킬 선택적 예외.

            delay_seconds:
                시간 초과 상황을 만들기 위한 호출 지연 시간.
        """
        self.output = output
        self.error = error
        self.delay_seconds = delay_seconds
        self.schema: type | None = None
        self.messages: list[dict[str, str]] | None = None

    def with_structured_output(self, schema: type) -> "FakeLlm":
        """요청받은 구조화 출력 모델을 기록하고 자신을 반환한다.

        Args:
            schema:
                LLM이 반환해야 하는 Pydantic 구조화 출력 모델.

        Returns:
            invoke를 이어서 호출할 현재 FakeLlm 인스턴스.
        """
        self.schema = schema
        return self

    def invoke(self, messages: list[dict[str, str]]) -> Any:
        """설정된 지연과 예외를 적용한 뒤 고정 결과를 반환한다.

        Args:
            messages:
                발화 생성에 전달된 system 및 user 메시지.

        Returns:
            생성 시 지정한 구조화 출력 대역값.

        Raises:
            Exception:
                생성 시 error가 지정된 경우 해당 예외.
        """
        self.messages = messages
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        return self.output


def make_runtime(llm: Any = None) -> SimpleNamespace:
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
        (utterance.question, "좋습니다. 다음 질문을 드리겠습니다."),
        (utterance.follow_up, "말씀해 주신 내용에서 한 가지를 조금 더 여쭤보겠습니다."),
        (
            utterance.challenge,
            "말씀하신 내용을 정확히 확인하기 위해 한 가지 더 질문드리겠습니다.",
        ),
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
    assert result["utterance_queue"] == [result["last_utterance"]]
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


def test_llm_preamble_uses_recent_transcript_and_last_signal():
    """LLM에는 최근 네 턴과 직전 평가 신호만 전달하고 결과를 발화에 사용한다."""
    question = make_question(QuestionKind.FOLLOW_UP)
    transcript = [
        Turn(role="candidate", text=f"대화-{index}", question_id=question.question_id)
        for index in range(6)
    ]
    llm = FakeLlm(output={"preamble": "좋습니다. 관련 내용을 조금 더 확인하겠습니다."})
    state = SessionState(
        session_id="session-llm",
        current_question=question,
        turn_type="question",
        transcript=transcript,
        last_signal={"quality": "bonus_available", "next_probe_target": "DI"},
    )

    result = compose_utterance(state, make_runtime(llm))

    assert llm.schema is ComposedUtterance
    assert llm.messages is not None
    user_prompt = llm.messages[1]["content"]
    assert "대화-0" not in user_prompt
    assert "대화-1" not in user_prompt
    assert "대화-2" in user_prompt
    assert "대화-5" in user_prompt
    assert "bonus_available" in user_prompt
    assert "next_probe_target" in user_prompt
    assert result["last_utterance"].startswith(
        "좋습니다. 관련 내용을 조금 더 확인하겠습니다."
    )
    assert result["last_utterance"].endswith(question.text)


@pytest.mark.parametrize(
    "llm",
    [
        FakeLlm(error=RuntimeError("LLM 호출 실패")),
        FakeLlm(output={"preamble": ""}),
        FakeLlm(output={"preamble": "FastAPI의 의존성 주입 방식에 대해 설명해 주세요."}),
    ],
)
def test_invalid_llm_result_falls_back_to_template(llm):
    """LLM 예외, 빈 출력, 질문 본문 반복은 모두 기본 템플릿으로 폴백한다.

    Args:
        llm:
            실패 또는 잘못된 결과를 반환하도록 설정한 FakeLlm.
    """
    question = make_question()
    state = SessionState(
        session_id="session-fallback",
        current_question=question,
        turn_type="question",
    )

    result = compose_utterance(state, make_runtime(llm))

    assert result["last_utterance"] == f"{utterance.question()}\n\n{question.text}"
    assert result["error"] is None


def test_llm_timeout_falls_back_to_template(monkeypatch):
    """LLM이 제한 시간을 넘기면 기다리지 않고 기본 템플릿으로 폴백한다.

    Args:
        monkeypatch:
            테스트 동안 LLM 제한 시간을 짧게 바꾸는 pytest fixture.
    """
    monkeypatch.setattr(interviewer_graph, "_UTTERANCE_LLM_TIMEOUT_SECONDS", 0.01)
    question = make_question()
    llm = FakeLlm(
        output={"preamble": "늦게 생성된 안내 문장입니다."},
        delay_seconds=0.1,
    )
    state = SessionState(
        session_id="session-timeout",
        current_question=question,
        turn_type="question",
    )

    result = compose_utterance(state, make_runtime(llm))

    assert result["last_utterance"] == f"{utterance.question()}\n\n{question.text}"
    assert result["error"] is None
