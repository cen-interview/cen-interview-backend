"""Interviewer LangGraph의 이벤트 및 평가 품질 분기 테스트."""

import uuid

import pytest
from langgraph.types import Command

from interview.interviewer.session import SessionState
from interview.interviewer.workflow.graph import get_compiled_graph
from interview.interviewer.workflow.runtime import InterviewDeps
from interview.schemas.question import Difficulty, Question, QuestionCategory, QuestionKind
from interview.schemas.report import FinalReport
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


def make_question(
    question_id: str,
    kind: QuestionKind = QuestionKind.MAIN,
    parent_question_id: str | None = None,
) -> Question:
    """그래프 흐름 테스트에서 사용할 질문을 만든다.

    Args:
        question_id:
            생성할 질문의 고유 ID.

        kind:
            생성할 질문의 종류.

        parent_question_id:
            파생 질문이 연결될 부모 질문 ID.

    Returns:
        지정한 종류와 부모 관계를 가진 Question.
    """
    return Question(
        question_id=question_id,
        text=f"{question_id} 테스트 질문입니다.",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=kind,
        category=QuestionCategory.TECHNICAL,
        parent_question_id=parent_question_id,
    )


class FakeStrategy:
    """생성한 질문 종류와 호출 횟수를 기록하는 Strategy fake."""

    def __init__(self) -> None:
        """메인 질문 순번과 Strategy 호출 기록을 초기화한다."""
        self.main_count = 0
        self.calls: list[str] = []

    def next_question(self, last_signal: AnswerQualitySignal | None) -> Question:
        """순번이 포함된 다음 메인 질문을 반환한다.

        Args:
            last_signal:
                직전 답변 평가 신호. 질문 내용에는 사용하지 않는다.

        Returns:
            새 메인 질문.
        """
        self.main_count += 1
        self.calls.append("next_question")
        return make_question(f"q-main-{self.main_count}")

    def _derived(
        self,
        call_name: str,
        kind: QuestionKind,
        parent_question_id: str,
    ) -> Question:
        """호출 기록을 남기고 부모 질문에 연결된 파생 질문을 만든다.

        Args:
            call_name:
                calls에 기록할 Strategy 메서드 이름.

            kind:
                생성할 파생 질문 종류.

            parent_question_id:
                파생 질문이 연결될 부모 질문 ID.

        Returns:
            부모 질문에 연결된 파생 Question.
        """
        self.calls.append(call_name)
        return make_question(
            question_id=f"q-{kind.value}-{len(self.calls)}",
            kind=kind,
            parent_question_id=parent_question_id,
        )

    def next_follow_up(self, topic, parent_question_id, target=None, answer_excerpt=None):
        """꼬리 질문을 반환한다."""
        return self._derived("next_follow_up", QuestionKind.FOLLOW_UP, parent_question_id)

    def next_challenge(self, topic, parent_question_id, target=None, answer_excerpt=None):
        """압박 질문을 반환한다."""
        return self._derived("next_challenge", QuestionKind.CHALLENGE, parent_question_id)

    def next_confirm_positive(self, topic, parent_question_id, target=None, answer_excerpt=None):
        """긍정 확인 질문을 반환한다."""
        return self._derived(
            "next_confirm_positive", QuestionKind.CONFIRM_POSITIVE, parent_question_id
        )

    def next_confirm_negative(self, topic, parent_question_id, target=None, answer_excerpt=None):
        """부정 확인 질문을 반환한다."""
        return self._derived(
            "next_confirm_negative", QuestionKind.CONFIRM_NEGATIVE, parent_question_id
        )

    def next_trap(self, topic, parent_question_id, target=None, answer_excerpt=None):
        """함정 질문을 반환한다."""
        return self._derived("next_trap", QuestionKind.TRAP, parent_question_id)


class FakeAssessment:
    """지정된 평가 품질을 순서대로 반환하는 Assessment fake."""

    def __init__(self, qualities: list[AnswerQuality]) -> None:
        """반환할 품질과 호출 기록을 초기화한다.

        Args:
            qualities:
                evaluate 호출 순서대로 반환할 답변 품질 목록.
        """
        self.qualities = qualities
        self.evaluate_calls: list[dict] = []
        self.completed_sets: list[str] = []
        self.finalize_count = 0

    def evaluate(self, question, answer_text, delivery_metrics=None) -> AnswerQualitySignal:
        """현재 호출 순서에 해당하는 고정 평가 신호를 반환한다."""
        call_index = len(self.evaluate_calls)
        quality = self.qualities[call_index]
        self.evaluate_calls.append({"question": question, "answer_text": answer_text})
        return AnswerQualitySignal(
            answer_id=f"answer-{call_index + 1}",
            question_id=question.question_id,
            quality=quality,
            next_probe_target="probe",
        )

    def complete_question_set(self, main_question_id: str) -> None:
        """완료 요청을 받은 메인 질문 ID를 기록한다."""
        self.completed_sets.append(main_question_id)

    def finalize(self) -> FinalReport:
        """호출 횟수를 기록하고 테스트용 최종 리포트를 반환한다."""
        self.finalize_count += 1
        return FinalReport(
            summary="테스트 면접 완료",
            overall_score=80.0,
            strengths=[],
            improvement_points=[],
            learning_recommendations=[],
            evaluations=[],
        )


def start_graph(qualities: list[AnswerQuality], **state_values):
    """고유 세션으로 그래프를 시작하고 테스트 실행 묶음을 반환한다.

    Args:
        qualities:
            FakeAssessment가 순서대로 반환할 품질 목록.

        **state_values:
            SessionState 생성 시 덮어쓸 선택적 값.

    Returns:
        graph, config, deps, session_id, 첫 interrupt 결과의 tuple.
    """
    session_id = f"session-{uuid.uuid4().hex}"
    strategy = FakeStrategy()
    assessment = FakeAssessment(qualities)
    deps = InterviewDeps(strategy=strategy, assessment=assessment)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = SessionState(session_id=session_id, **state_values)
    graph = get_compiled_graph()
    result = graph.invoke(state, config=config, context=deps)
    return graph, config, deps, session_id, result


def resume(graph, config, deps, event: dict):
    """현재 checkpoint에 이벤트를 전달해 그래프를 재개한다."""
    return graph.invoke(
        Command(resume={"event": event, "delivery_metrics": None}),
        config=config,
        context=deps,
    )


def answer_event(session_id: str, question_id: str, text: str = "테스트 답변") -> dict:
    """현재 질문에 대한 답변 제출 이벤트 dict를 만든다."""
    return {
        "type": "answer_submitted",
        "session_id": session_id,
        "question_id": question_id,
        "text": text,
    }


@pytest.mark.parametrize(
    ("quality", "expected_kind"),
    [
        (AnswerQuality.SUFFICIENT, QuestionKind.MAIN),
        (AnswerQuality.BONUS_AVAILABLE, QuestionKind.FOLLOW_UP),
        (AnswerQuality.MISCONCEPTION, QuestionKind.CHALLENGE),
        (AnswerQuality.CONFIRM_POSITIVE, QuestionKind.CONFIRM_POSITIVE),
        (AnswerQuality.CONFIRM_NEGATIVE, QuestionKind.CONFIRM_NEGATIVE),
        (AnswerQuality.TRAP_AVAILABLE, QuestionKind.TRAP),
    ],
)
def test_quality_routes_to_expected_question_kind(quality, expected_kind):
    """여섯 가지 평가 품질이 올바른 종류의 다음 질문을 만든다."""
    graph, config, deps, session_id, result = start_graph([quality])
    question = result["current_question"]

    result = resume(graph, config, deps, answer_event(session_id, question.question_id))

    assert result["current_question"].kind == expected_kind
    assert result["last_utterance"].endswith(result["current_question"].text)
    assert result["utterance_queue"] == [result["last_utterance"]]
    assert result["transcript"][-1].role == "interviewer"
    assert result["transcript"][-1].question_id == result["current_question"].question_id


def test_first_question_is_composed_and_recorded_before_interrupt():
    """첫 질문은 시작 인사와 조립되어 큐와 transcript에 기록된 뒤 대기한다."""
    _, _, _, _, result = start_graph([AnswerQuality.SUFFICIENT])
    question = result["current_question"]

    assert "__interrupt__" in result
    assert result["last_utterance"].startswith("안녕하세요.")
    assert result["last_utterance"].endswith(question.text)
    assert result["utterance_queue"] == [result["last_utterance"]]
    assert result["transcript"][-1].role == "interviewer"
    assert result["transcript"][-1].question_id == question.question_id


def test_challenge_is_not_created_twice_in_same_question_set():
    """같은 질문 세트에서 두 번째 misconception은 새 challenge를 만들지 않는다."""
    graph, config, deps, session_id, result = start_graph(
        [AnswerQuality.MISCONCEPTION, AnswerQuality.MISCONCEPTION],
        max_derived_turns_per_set=3,
    )
    first = result["current_question"]
    result = resume(graph, config, deps, answer_event(session_id, first.question_id))
    challenge = result["current_question"]
    result = resume(graph, config, deps, answer_event(session_id, challenge.question_id))

    assert result["current_question"].kind == QuestionKind.MAIN
    assert deps.strategy.calls.count("next_challenge") == 1


def test_derived_question_limit_completes_current_set():
    """최대 파생 질문 수에 도달하면 추가 파생 질문 없이 다음 메인 질문으로 간다."""
    qualities = [
        AnswerQuality.BONUS_AVAILABLE,
        AnswerQuality.CONFIRM_POSITIVE,
        AnswerQuality.BONUS_AVAILABLE,
    ]
    graph, config, deps, session_id, result = start_graph(qualities)

    for _ in qualities:
        question = result["current_question"]
        result = resume(graph, config, deps, answer_event(session_id, question.question_id))

    assert result["current_question"].kind == QuestionKind.MAIN
    assert result["derived_turn_count"] == 0
    assert deps.strategy.calls.count("next_follow_up") == 1


def test_wrong_question_id_does_not_call_assessment():
    """현재 질문과 다른 question_id의 답변은 평가하지 않고 질문을 유지한다."""
    graph, config, deps, session_id, result = start_graph([AnswerQuality.SUFFICIENT])
    current_question = result["current_question"]

    result = resume(graph, config, deps, answer_event(session_id, "wrong-question"))

    assert result["current_question"] == current_question
    assert deps.assessment.evaluate_calls == []
    assert result["asked_count"] == 1


def test_replay_does_not_increase_asked_count():
    """다시 듣기 이벤트는 현재 질문과 메인 질문 수를 변경하지 않는다."""
    graph, config, deps, session_id, result = start_graph([AnswerQuality.SUFFICIENT])
    current_question = result["current_question"]

    result = resume(
        graph,
        config,
        deps,
        {"type": "replay_requested", "session_id": session_id},
    )

    assert result["current_question"] == current_question
    assert result["asked_count"] == 1
    assert deps.assessment.evaluate_calls == []
    assert result["last_utterance"].startswith("네, 질문을 다시 말씀드리겠습니다.")
    assert result["last_utterance"].endswith(current_question.text)
    assert result["utterance_queue"] == [result["last_utterance"]]
    assert result["transcript"][-1].role == "interviewer"


def test_end_event_creates_report_and_finishes_session():
    """명시적인 종료 이벤트는 최종 리포트를 만들고 세션을 종료한다."""
    graph, config, deps, session_id, _ = start_graph([AnswerQuality.SUFFICIENT])

    result = resume(
        graph,
        config,
        deps,
        {"type": "end_requested", "session_id": session_id},
    )

    assert result["finished"] is True
    assert result["report"]["summary"] == "테스트 면접 완료"
    assert deps.assessment.finalize_count == 1
    assert result["last_utterance"] == "이상으로 면접을 마치겠습니다. 참여해 주셔서 감사합니다."
    assert result["utterance_queue"] == [result["last_utterance"]]
    assert result["transcript"][-1].role == "interviewer"
    assert result["transcript"][-1].question_id is None


def test_ten_main_questions_are_all_evaluated_before_finish():
    """최대 질문 수가 10이면 마지막 질문을 포함해 답변을 정확히 10회 평가한다."""
    graph, config, deps, session_id, result = start_graph(
        [AnswerQuality.SUFFICIENT] * 10,
        max_questions=10,
    )

    for _ in range(10):
        question = result["current_question"]
        result = resume(graph, config, deps, answer_event(session_id, question.question_id))

    assert len(deps.assessment.evaluate_calls) == 10
    assert len(deps.assessment.completed_sets) == 10
    assert result["asked_count"] == 10
    assert result["finished"] is True
