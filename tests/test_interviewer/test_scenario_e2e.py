"""Fake 에이전트로 Interviewer의 10단계 통합 시나리오를 검증한다.

단일 노드나 라우터의 반환값보다 실제 LangGraph를 시작하고 이벤트를 연속으로
resume했을 때 질문 세트, 침묵 정책, 종료 상태가 끝까지 이어지는지를 확인한다.
외부 LLM이나 실제 Strategy/Assessment 구현에는 의존하지 않는다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

from interview.interviewer.facade import InterviewSession
from interview.interviewer.session import SessionState
from interview.interviewer.workflow.graph import get_compiled_graph
from interview.interviewer.workflow.runtime import InterviewDeps
from interview.schemas.events import EndRequested, Mode, ReplayRequested
from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Difficulty, Question, QuestionCategory, QuestionKind
from interview.schemas.report import FinalReport
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


def make_question(
    question_id: str,
    kind: QuestionKind = QuestionKind.MAIN,
    parent_question_id: str | None = None,
) -> Question:
    """통합 시나리오에서 사용할 고정 질문을 만든다.

    Args:
        question_id:
            생성할 질문의 고유 ID.

        kind:
            메인, 꼬리, 압박, 힌트 등 질문의 역할.

        parent_question_id:
            파생 질문이 연결될 직전 질문 ID. 메인 질문과 힌트는 None을 사용한다.

    Returns:
        FastAPI 주제와 고정 Evidence ID를 가진 테스트용 Question.
    """
    return Question(
        question_id=question_id,
        text=f"{question_id} 테스트 질문입니다.",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=kind,
        category=QuestionCategory.TECHNICAL,
        parent_question_id=parent_question_id,
        evidence_ids=["evidence-fastapi"],
    )


class ScenarioStrategy:
    """질문 생성 호출을 기록하고 결정적인 질문을 반환하는 Strategy fake."""

    def __init__(self) -> None:
        """메인 질문 순번과 메서드 호출 기록을 초기화한다."""
        self.main_count = 0
        self.calls: list[str] = []

    def next_question(self, last_signal: AnswerQualitySignal | None) -> Question:
        """순번이 포함된 다음 메인 질문을 반환한다.

        Args:
            last_signal:
                직전 답변의 평가 신호. Fake 질문 내용에는 사용하지 않는다.

        Returns:
            호출 순번으로 식별할 수 있는 새 메인 질문.
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
        """Strategy 호출을 기록하고 부모 질문에 연결된 파생 질문을 만든다.

        Args:
            call_name:
                호출 기록에 저장할 Strategy 메서드 이름.

            kind:
                생성할 파생 질문 종류.

            parent_question_id:
                파생 질문이 이어지는 직전 질문 ID.

        Returns:
            호출 순번과 종류가 ID에 포함된 파생 Question.
        """
        self.calls.append(call_name)
        return make_question(
            question_id=f"q-{kind.value}-{len(self.calls)}",
            kind=kind,
            parent_question_id=parent_question_id,
        )

    def next_follow_up(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        """BONUS 신호에 사용할 꼬리 질문을 반환한다.

        Args:
            topic:
                현재 질문의 주제. Fake에서는 사용하지 않는다.

            parent_question_id:
                꼬리 질문이 이어지는 직전 질문 ID.

            target:
                추가로 확인할 대상. Fake에서는 사용하지 않는다.

            answer_excerpt:
                질문 생성에 참고할 답변 일부. Fake에서는 사용하지 않는다.

        Returns:
            직전 질문에 연결된 FOLLOW_UP 질문.
        """
        return self._derived("next_follow_up", QuestionKind.FOLLOW_UP, parent_question_id)

    def next_challenge(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        """MISCONCEPTION 신호에 사용할 압박 질문을 반환한다.

        Args:
            topic:
                현재 질문의 주제. Fake에서는 사용하지 않는다.

            parent_question_id:
                압박 질문이 이어지는 직전 질문 ID.

            target:
                오개념을 추가 확인할 대상. Fake에서는 사용하지 않는다.

            answer_excerpt:
                질문 생성에 참고할 답변 일부. Fake에서는 사용하지 않는다.

        Returns:
            직전 질문에 연결된 CHALLENGE 질문.
        """
        return self._derived("next_challenge", QuestionKind.CHALLENGE, parent_question_id)

    def next_confirm_positive(
        self,
        topic: str,
        parent_question_id: str,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        """파생 질문 제한 시나리오에 사용할 긍정 확인 질문을 반환한다.

        Args:
            topic:
                현재 질문의 주제. Fake에서는 사용하지 않는다.

            parent_question_id:
                확인 질문이 이어지는 직전 질문 ID.

            target:
                추가로 확인할 대상. Fake에서는 사용하지 않는다.

            answer_excerpt:
                질문 생성에 참고할 답변 일부. Fake에서는 사용하지 않는다.

        Returns:
            직전 질문에 연결된 CONFIRM_POSITIVE 질문.
        """
        return self._derived(
            "next_confirm_positive",
            QuestionKind.CONFIRM_POSITIVE,
            parent_question_id,
        )

    def next_hint(
        self,
        question: Question,
        target: str | None = None,
        answer_excerpt: str | None = None,
    ) -> Question:
        """첫 번째 유효 침묵에 제시할 힌트 질문을 반환한다.

        Args:
            question:
                침묵이 발생한 현재 질문.

            target:
                힌트가 집중할 대상. Fake에서는 사용하지 않는다.

            answer_excerpt:
                질문 생성에 참고할 답변 일부. 침묵 시나리오에서는 None이다.

        Returns:
            원래 질문의 ID를 문구에 포함한 HINT 질문.
        """
        self.calls.append("next_hint")
        return Question(
            question_id=f"q-hint-{len(self.calls)}",
            text=f"힌트: {question.question_id}의 핵심 개념부터 설명해 보세요.",
            topic=question.topic,
            difficulty=question.difficulty,
            kind=QuestionKind.HINT,
            category=question.category,
            evidence_ids=question.evidence_ids,
        )


class ScenarioAssessment:
    """정해진 답변 품질을 순서대로 반환하는 Assessment fake."""

    def __init__(self, qualities: list[AnswerQuality]) -> None:
        """평가 품질 시나리오와 호출 기록을 초기화한다.

        Args:
            qualities:
                evaluate 호출 순서대로 반환할 AnswerQuality 목록.
        """
        self.qualities = qualities
        self.evaluate_calls: list[dict[str, Any]] = []
        self.completed_sets: list[str] = []
        self.finalize_count = 0

    def evaluate(
        self,
        question: Question,
        answer_text: str,
        delivery_metrics: dict[str, Any] | None = None,
    ) -> AnswerQualitySignal:
        """현재 호출 순서에 지정된 답변 품질 신호를 반환한다.

        Args:
            question:
                현재 지원자가 답변한 질문.

            answer_text:
                지원자가 제출한 답변 본문.

            delivery_metrics:
                선택적인 음성 전달 지표.

        Returns:
            질문 ID와 예정된 품질이 담긴 AnswerQualitySignal.

        Raises:
            AssertionError:
                시나리오에 준비된 횟수보다 evaluate가 더 많이 호출된 경우.
        """
        call_index = len(self.evaluate_calls)
        if call_index >= len(self.qualities):
            raise AssertionError("시나리오에 준비되지 않은 추가 평가가 호출되었습니다.")

        self.evaluate_calls.append(
            {
                "question": question,
                "answer_text": answer_text,
                "delivery_metrics": delivery_metrics,
            }
        )
        return AnswerQualitySignal(
            answer_id=f"answer-{call_index + 1}",
            question_id=question.question_id,
            quality=self.qualities[call_index],
            next_probe_target="FastAPI 핵심 개념",
        )

    def complete_question_set(self, main_question_id: str) -> None:
        """완료된 메인 질문 세트의 ID를 기록한다.

        Args:
            main_question_id:
                평가 완료 처리를 요청받은 기준 메인 질문 ID.
        """
        self.completed_sets.append(main_question_id)

    def finalize(self) -> FinalReport:
        """호출 횟수를 기록하고 고정된 최종 리포트를 반환한다.

        Returns:
            종료 시나리오에서 식별할 수 있는 테스트용 FinalReport.
        """
        self.finalize_count += 1
        return FinalReport(
            summary="10-1 통합 시나리오 완료",
            overall_score=80.0,
            strengths=[],
            improvement_points=[],
            learning_recommendations=[],
            evaluations=[],
        )


@dataclass
class ScenarioRunner:
    """하나의 checkpoint thread에서 이벤트를 연속 실행하는 테스트 도우미.

    Attributes:
        graph:
            Interviewer의 compiled LangGraph.

        config:
            시나리오 전용 thread_id가 들어 있는 LangGraph 설정.

        deps:
            Fake Strategy와 Assessment가 담긴 runtime context.

        session_id:
            이벤트와 상태가 공유하는 면접 세션 ID.

        state:
            마지막 interrupt 또는 END까지 실행된 그래프 결과.

        strategy:
            질문 생성 호출을 확인할 ScenarioStrategy.

        assessment:
            평가와 질문 세트 완료 호출을 확인할 ScenarioAssessment.
    """

    graph: Any
    config: dict[str, dict[str, str]]
    deps: InterviewDeps
    session_id: str
    state: dict[str, Any]
    strategy: ScenarioStrategy
    assessment: ScenarioAssessment

    def submit(self, event: dict[str, Any]) -> dict[str, Any]:
        """현재 interrupt에 이벤트를 전달하고 다음 대기점까지 실행한다.

        Args:
            event:
                resume payload에 넣을 직렬화 가능한 Interviewer 이벤트.

        Returns:
            이벤트 처리 후 다음 interrupt 또는 END에 도달한 최신 그래프 상태.
        """
        self.state = self.graph.invoke(
            Command(resume={"event": event, "delivery_metrics": None}),
            config=self.config,
            context=self.deps,
        )
        return self.state

    def answer(self, text: str = "테스트 답변") -> dict[str, Any]:
        """현재 질문에 대한 답변을 제출한다.

        Args:
            text:
                Assessment fake에 전달할 지원자 답변.

        Returns:
            답변 평가와 품질 분기까지 끝난 최신 그래프 상태.

        Raises:
            AssertionError:
                현재 상태에 답변할 질문이 없는 경우.
        """
        question = self.state.get("current_question")
        if question is None:
            raise AssertionError("답변을 제출할 현재 질문이 없습니다.")
        return self.submit(
            {
                "type": "answer_submitted",
                "session_id": self.session_id,
                "question_id": question.question_id,
                "text": text,
            }
        )

    def silence(self, duration_seconds: float = 8.0) -> dict[str, Any]:
        """현재 질문 대기 상태에 침묵 감지 이벤트를 전달한다.

        Args:
            duration_seconds:
                감지된 침묵 지속 시간. 기본값은 정책 임계값과 같다.

        Returns:
            침묵 정책의 다음 행동까지 반영된 최신 그래프 상태.
        """
        return self.submit(
            {
                "type": "silence_detected",
                "session_id": self.session_id,
                "silence_duration_seconds": duration_seconds,
            }
        )


def start_scenario(
    qualities: list[AnswerQuality],
    **state_values: Any,
) -> ScenarioRunner:
    """고유 세션과 Fake 의존성으로 Interviewer 그래프를 시작한다.

    Args:
        qualities:
            ScenarioAssessment가 evaluate 호출 순서대로 반환할 답변 품질 목록.

        **state_values:
            max_questions나 파생 질문 제한처럼 초기 SessionState에서 덮어쓸 값.

    Returns:
        첫 질문 발화 후 interrupt에서 대기 중인 ScenarioRunner.
    """
    session_id = f"scenario-{uuid.uuid4().hex}"
    strategy = ScenarioStrategy()
    assessment = ScenarioAssessment(qualities)
    deps = InterviewDeps(strategy=strategy, assessment=assessment)
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke(
        SessionState(session_id=session_id, **state_values),
        config=config,
        context=deps,
    )
    return ScenarioRunner(
        graph=graph,
        config=config,
        deps=deps,
        session_id=session_id,
        state=state,
        strategy=strategy,
        assessment=assessment,
    )


def test_sufficient_answer_advances_to_next_main_question() -> None:
    """충분한 답변은 현재 세트를 완료하고 다음 메인 질문으로 이동한다."""
    scenario = start_scenario([AnswerQuality.SUFFICIENT])

    result = scenario.answer()

    assert scenario.assessment.completed_sets == ["q-main-1"]
    assert result["current_question"].question_id == "q-main-2"
    assert result["current_question"].kind == QuestionKind.MAIN
    assert result["asked_count"] == 2


def test_bonus_follow_up_then_sufficient_answer_completes_set() -> None:
    """BONUS 꼬리 질문의 답변까지 평가한 뒤 원래 메인 질문 세트를 완료한다."""
    scenario = start_scenario(
        [AnswerQuality.BONUS_AVAILABLE, AnswerQuality.SUFFICIENT]
    )

    follow_up_state = scenario.answer("메인 질문 답변")
    follow_up = follow_up_state["current_question"]
    assert follow_up.kind == QuestionKind.FOLLOW_UP
    assert follow_up.parent_question_id == "q-main-1"

    result = scenario.answer("꼬리 질문 답변")

    assert scenario.strategy.calls.count("next_follow_up") == 1
    assert scenario.assessment.completed_sets == ["q-main-1"]
    assert result["current_question"].question_id == "q-main-2"
    assert result["derived_turn_count"] == 0


def test_misconception_uses_one_challenge_then_completes_set() -> None:
    """같은 질문 세트의 두 번째 오개념은 challenge를 반복하지 않고 세트를 끝낸다."""
    scenario = start_scenario(
        [AnswerQuality.MISCONCEPTION, AnswerQuality.MISCONCEPTION],
        max_derived_turns_per_set=3,
    )

    challenge_state = scenario.answer("오개념이 포함된 답변")
    challenge = challenge_state["current_question"]
    assert challenge.kind == QuestionKind.CHALLENGE
    assert challenge.parent_question_id == "q-main-1"

    result = scenario.answer("압박 질문에도 남아 있는 오개념")

    assert scenario.strategy.calls.count("next_challenge") == 1
    assert scenario.assessment.completed_sets == ["q-main-1"]
    assert result["current_question"].question_id == "q-main-2"
    assert result["challenge_used_in_set"] is False


def test_derived_question_limit_completes_set() -> None:
    """파생 질문 제한에 도달하면 추가 파생 질문 없이 다음 메인 질문으로 이동한다."""
    scenario = start_scenario(
        [
            AnswerQuality.BONUS_AVAILABLE,
            AnswerQuality.CONFIRM_POSITIVE,
            AnswerQuality.BONUS_AVAILABLE,
        ],
        max_derived_turns_per_set=2,
    )

    scenario.answer("꼬리 질문이 필요한 답변")
    scenario.answer("확인이 더 필요한 답변")
    result = scenario.answer("파생 질문 제한에 도달한 답변")

    assert scenario.strategy.calls.count("next_follow_up") == 1
    assert scenario.strategy.calls.count("next_confirm_positive") == 1
    assert scenario.assessment.completed_sets == ["q-main-1"]
    assert result["current_question"].question_id == "q-main-2"
    assert result["derived_turn_count"] == 0


def test_silence_progresses_through_hint_replay_and_timeout() -> None:
    """세 번의 유효 침묵은 힌트, 재제시, 종료 타임아웃 순서로 처리된다."""
    scenario = start_scenario([])

    hint_state = scenario.silence()
    hint_question = hint_state["current_question"]
    assert hint_state["silence_action"] == "hint"
    assert hint_state["silence_count"] == 1
    assert hint_question.kind == QuestionKind.HINT
    assert scenario.strategy.calls.count("next_hint") == 1

    replay_state = scenario.silence()
    assert replay_state["silence_action"] == "replay"
    assert replay_state["silence_count"] == 2
    assert replay_state["current_question"] == hint_question
    assert replay_state["asked_count"] == 1

    result = scenario.silence()

    assert result["silence_action"] == "timeout"
    assert result["silence_count"] == 3
    assert result["timeout_action"] == "end"
    assert result["finished"] is True
    assert result["report"]["summary"] == "10-1 통합 시나리오 완료"
    assert scenario.assessment.evaluate_calls == []
    assert scenario.assessment.finalize_count == 1


def test_replay_keeps_current_question_and_main_question_count() -> None:
    """다시 듣기는 현재 질문과 메인 질문 수를 유지하고 답변을 평가하지 않는다."""
    scenario = start_scenario([])
    current_question = scenario.state["current_question"]

    result = scenario.submit(
        {
            "type": "replay_requested",
            "session_id": scenario.session_id,
            "question_id": current_question.question_id,
        }
    )

    assert result["current_question"] == current_question
    assert result["asked_count"] == 1
    assert result["turn_type"] == "replay"
    assert scenario.assessment.evaluate_calls == []


def test_end_request_returns_same_finished_state_for_later_events() -> None:
    """종료된 InterviewSession은 후속 이벤트를 실행하지 않고 같은 종료 상태를 반환한다."""
    session_id = f"scenario-facade-{uuid.uuid4().hex}"
    strategy = ScenarioStrategy()
    assessment = ScenarioAssessment([])
    session = InterviewSession(
        session_id=session_id,
        mode=Mode.CHAT,
        coverage=CoverageMap(),
        deps=InterviewDeps(strategy=strategy, assessment=assessment),
    )
    session.start_session()

    finished_state = session.submit_event(EndRequested(session_id=session_id))
    repeated_state = session.submit_event(ReplayRequested(session_id=session_id))

    assert finished_state.finished is True
    assert finished_state.report is not None
    assert finished_state.report["summary"] == "10-1 통합 시나리오 완료"
    assert repeated_state.model_dump(mode="json") == finished_state.model_dump(mode="json")
    assert assessment.finalize_count == 1


def test_last_main_question_is_evaluated_before_session_finishes() -> None:
    """최대 메인 질문의 마지막 답변과 질문 세트까지 처리한 뒤 리포트를 생성한다."""
    scenario = start_scenario(
        [AnswerQuality.SUFFICIENT] * 3,
        max_questions=3,
    )

    for answer_number in range(1, 4):
        scenario.answer(f"{answer_number}번째 메인 질문 답변")

    result = scenario.state
    candidate_turns = [
        turn for turn in result["transcript"] if turn.role == "candidate"
    ]

    assert len(scenario.assessment.evaluate_calls) == 3
    assert scenario.assessment.completed_sets == ["q-main-1", "q-main-2", "q-main-3"]
    assert len(candidate_turns) == 3
    assert result["asked_count"] == 3
    assert result["finished"] is True
    assert result["report"]["summary"] == "10-1 통합 시나리오 완료"
    assert scenario.assessment.finalize_count == 1
