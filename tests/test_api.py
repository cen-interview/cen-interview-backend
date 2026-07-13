"""실제 Strategy와 FakeAssessment를 사용하는 채팅 API 통합 시나리오."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from interview.api.main import app, get_interview_session_factory
from interview.interviewer.facade import create_session
from interview.interviewer.workflow.runtime import InterviewDeps
from interview.schemas.evidence import CoverageMap
from interview.schemas.events import Mode
from interview.schemas.report import FinalReport
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal
from interview.strategy import StrategyAgent


class FakeAssessment:
    """채팅 API 흐름에 집중하기 위해 답변을 항상 충분하다고 평가하는 fake."""

    def __init__(self) -> None:
        """평가, 질문 세트 완료, 리포트 생성 호출 내역을 초기화한다."""
        self.evaluate_calls: list[dict] = []
        self.completed_sets: list[str] = []
        self.finalize_count = 0

    def evaluate(
        self,
        question,
        answer_text: str,
        delivery_metrics: dict | None = None,
    ) -> AnswerQualitySignal:
        """답변을 기록하고 다음 메인 질문으로 진행할 평가 신호를 반환한다.

        Args:
            question:
                실제 StrategyAgent가 생성한 현재 질문.

            answer_text:
                채팅 API로 제출된 지원자 답변.

            delivery_metrics:
                채팅 모드에서는 None인 선택적 전달 지표.

        Returns:
            quality가 sufficient인 AnswerQualitySignal.
        """
        self.evaluate_calls.append(
            {
                "question": question,
                "answer_text": answer_text,
                "delivery_metrics": delivery_metrics,
            }
        )
        return AnswerQualitySignal(
            answer_id=f"fake-answer-{len(self.evaluate_calls)}",
            question_id=question.question_id,
            quality=AnswerQuality.SUFFICIENT,
            rationale=["채팅 API 흐름 확인용 고정 평가"],
            accuracy=1.0,
            sufficiency=1.0,
        )

    def complete_question_set(self, main_question_id: str) -> None:
        """완료된 메인 질문 세트 ID를 기록한다.

        Args:
            main_question_id:
                그래프가 완료 처리한 메인 질문 ID.
        """
        self.completed_sets.append(main_question_id)

    def finalize(self) -> FinalReport:
        """호출 횟수를 기록하고 테스트용 최종 리포트를 반환한다.

        Returns:
            API의 종료 응답 형식을 확인할 수 있는 고정 FinalReport.
        """
        self.finalize_count += 1
        return FinalReport(
            summary="실제 Strategy와 FakeAssessment 채팅 면접 완료",
            overall_score=100.0,
            strengths=["채팅 면접 흐름을 완료했습니다."],
            improvement_points=[],
            learning_recommendations=[],
            evaluations=[],
        )


@pytest.fixture
def chat_api() -> Iterator[tuple[TestClient, list[FakeAssessment]]]:
    """실제 Strategy와 FakeAssessment를 사용하는 API client를 제공한다.

    세션마다 실제 StrategyAgent와 새로운 FakeAssessment를 생성한다. 최대 메인
    질문 수는 2로 줄여 마지막 질문까지 평가된 뒤 종료되는지 짧게 확인한다.

    Yields:
        TestClient와 생성된 FakeAssessment 목록의 tuple.
    """
    assessments: list[FakeAssessment] = []

    def fake_session_factory(mode: Mode):
        """실제 Strategy와 FakeAssessment를 조립해 테스트 세션을 만든다.

        Args:
            mode:
                API 요청에서 검증된 면접 모드.

        Returns:
            Registry에 등록된 InterviewSession과 첫 질문.
        """
        assessment = FakeAssessment()
        assessments.append(assessment)
        deps = InterviewDeps(
            strategy=StrategyAgent(CoverageMap()),
            assessment=assessment,
        )
        return create_session(
            mode=mode,
            max_questions=2,
            deps=deps,
        )

    app.dependency_overrides[get_interview_session_factory] = (
        lambda: fake_session_factory
    )
    with TestClient(app) as client:
        yield client, assessments
    app.dependency_overrides.pop(get_interview_session_factory, None)


def _start_chat_session(client: TestClient) -> dict:
    """채팅 세션을 생성하고 성공 응답 JSON을 반환한다.

    Args:
        client:
            FastAPI 테스트 client.

    Returns:
        첫 질문과 세션 상태가 담긴 응답 JSON.
    """
    response = client.post("/api/sessions", json={"mode": "chat"})
    assert response.status_code == 200
    return response.json()


def _submit_chat_answer(client: TestClient, session_id: str, text: str) -> dict:
    """채팅 답변 제출 이벤트를 보내고 응답 JSON을 반환한다.

    Args:
        client:
            FastAPI 테스트 client.

        session_id:
            답변을 제출할 면접 세션 ID.

        text:
            제출할 채팅 답변 본문.

    Returns:
        이벤트 처리 후 세션 상태가 담긴 응답 JSON.
    """
    response = client.post(
        f"/api/sessions/{session_id}/events",
        json={"payload": {"action": "submit", "text": text}},
    )
    assert response.status_code == 200
    return response.json()


def test_chat_runs_until_last_main_answer_and_returns_report(chat_api):
    """두 번째 메인 질문까지 평가한 뒤 채팅 세션과 리포트가 종료된다."""
    client, assessments = chat_api
    started = _start_chat_session(client)
    session_id = started["session_id"]
    first_question_id = started["question"]["question_id"]

    assert started["finished"] is False
    assert started["last_utterance"]
    assert started["utterance_queue"] == [started["last_utterance"]]
    assert started["turn_type"] == "greeting"
    assert started["transcript"][-1]["role"] == "interviewer"

    after_first = _submit_chat_answer(
        client,
        session_id,
        "첫 번째 질문에 대한 테스트 답변입니다.",
    )

    assert after_first["finished"] is False
    assert after_first["question"] is not None
    assert after_first["question"]["question_id"] != first_question_id

    finished = _submit_chat_answer(
        client,
        session_id,
        "두 번째 질문에 대한 테스트 답변입니다.",
    )

    assessment = assessments[0]
    assert finished["finished"] is True
    assert finished["question"] is None
    assert finished["turn_type"] == "closing"
    assert finished["report"]["summary"] == (
        "실제 Strategy와 FakeAssessment 채팅 면접 완료"
    )
    assert len(assessment.evaluate_calls) == 2
    assert len(assessment.completed_sets) == 2
    assert assessment.finalize_count == 1

    repeated = client.post(
        f"/api/sessions/{session_id}/events",
        json={"payload": {"action": "submit", "text": "종료 후 중복 답변"}},
    )

    assert repeated.status_code == 200
    assert repeated.json() == finished
    assert len(assessment.evaluate_calls) == 2
    assert assessment.finalize_count == 1


def test_chat_replay_keeps_question_and_end_returns_report(chat_api):
    """다시 듣기는 질문을 유지하고 명시적 종료는 리포트를 반환한다."""
    client, assessments = chat_api
    started = _start_chat_session(client)
    session_id = started["session_id"]
    question_id = started["question"]["question_id"]

    replay_response = client.post(
        f"/api/sessions/{session_id}/events",
        json={"payload": {"action": "replay"}},
    )

    assert replay_response.status_code == 200
    replayed = replay_response.json()
    assert replayed["finished"] is False
    assert replayed["question"]["question_id"] == question_id
    assert replayed["turn_type"] == "replay"
    assert assessments[0].evaluate_calls == []

    end_response = client.post(
        f"/api/sessions/{session_id}/events",
        json={"payload": {"action": "end"}},
    )

    assert end_response.status_code == 200
    ended = end_response.json()
    assert ended["finished"] is True
    assert ended["turn_type"] == "closing"
    assert ended["report"] is not None
    assert assessments[0].finalize_count == 1


def test_chat_event_returns_404_for_unknown_session(chat_api):
    """등록되지 않은 세션 ID의 이벤트 요청은 404를 반환한다."""
    client, _ = chat_api

    response = client.post(
        "/api/sessions/missing-session/events",
        json={"payload": {"action": "submit", "text": "답변"}},
    )

    assert response.status_code == 404
