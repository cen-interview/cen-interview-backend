"""Interviewer LangGraph에서 상태를 변경하거나 외부 의존성을 호출하는 노드."""

import logging
from typing import Any

from interview.interviewer.intent import detect_voice_command
from interview.assessment.rubric_store import get_rubric_store
from interview.interviewer.models import DeliveryMetrics
from interview.interviewer.workflow.runtime import (
    _restore_signal,
    _runtime_deps,
    _serialize_signal,
    _state_get,
)
from interview.interviewer.session import SessionState, SilencePolicy, TimeoutPolicy, Turn
from interview.schemas.events import (
    AnswerSubmitted,
    EndRequested,
    InterviewerEvent,
    ReplayRequested,
    SilenceDetected,
)
from interview.schemas.question import Question
from interview.schemas.report import ReportGenerationResult
from interview.schemas.rubric import RubricCandidate, RubricSource
from langgraph.types import interrupt
from pydantic import TypeAdapter, ValidationError

_EVENT_ADAPTER = TypeAdapter(InterviewerEvent)
_logger = logging.getLogger("uvicorn.error")

def greet(state: SessionState, runtime: Any) -> dict[str, Any]:
    """첫 메인 질문을 생성한다.

    3-2 skeleton의 시작 노드다. 첫 질문을 만들고 SessionState의 질문 진행
    필드를 초기화한다.
    """
    deps = _runtime_deps(runtime)
    question = deps.strategy.next_question(last_signal=None)

    return {
        "current_question": question,
        "asked_count": 1,
        "main_question_id": question.question_id,
        "main_topic": question.topic,
        "turn_type": "greeting",
        "finished": False,
        "error": None,
    }


def wait_event(state: SessionState, runtime: Any) -> dict[str, Any]:
    """지원자 입력을 기다리고 resume payload를 pending 필드에 저장한다.

    interrupt가 있는 노드는 재개 시 처음부터 다시 실행되므로, 여기에는
    다른 부작용을 두지 않는다. delivery_metrics는 이 단계에서 이벤트 안에
    합치지 않고 별도 pending 필드에 보관하며, 다음 validate_event가 음성 답변
    제출에만 사용할 수 있도록 정규화한다.
    """
    payload = interrupt({"waiting_for": "candidate"})

    return {
        "pending_event": payload["event"],
        "pending_delivery_metrics": payload.get("delivery_metrics"),
    }


def validate_event(state: SessionState | dict[str, Any]) -> dict[str, Any]:
    """대기 중인 이벤트를 복원하고 현재 세션에서 처리할 수 있는지 검증한다.

    체크포인터에 dict로 저장된 pending_event를 InterviewerEvent 타입으로
    복원하여 지원하는 이벤트인지 확인한다. 이어서 세션 ID를 검증하고, 음성
    세션의 답변 제출 이벤트라면 짧은 종료/다시 듣기 명령을 해당 이벤트로
    변환한다. 일반 답변은 현재 질문 ID와 빈 답변 여부도 확인한다.

    delivery_metrics는 음성 AnswerSubmitted에만 허용한다. 채팅 답변이나 다시
    듣기, 종료, 침묵, 타임아웃 이벤트에 함께 들어온 지표는 제거한다. 선택적인
    전달 지표 형식이 잘못된 경우에도 답변 내용 평가를 막지 않고 지표만 버린다.

    검증 실패는 예외로 그래프를 중단하지 않고 error에 사용자가 이해할 수 있는
    메시지를 저장한다. 성공한 이벤트와 전달 지표도 Pydantic 객체 자체를 상태에
    넣지 않고 JSON 직렬화가 가능한 dict로 다시 변환한다.

    Args:
        state:
            pending_event와 현재 세션 정보를 가진 SessionState 또는 같은 필드를
            가진 dict.

    Returns:
        검증에 성공하면 정규화된 pending_event, 음성 답변에만 허용된
        pending_delivery_metrics와 error=None을 담은 부분 상태. 실패하면 원인을
        설명하는 error를 담은 부분 상태.
    """
    pending_event = _state_get(state, "pending_event")
    if pending_event is None:
        return {"error": "처리할 이벤트가 없습니다."}

    try:
        event = _EVENT_ADAPTER.validate_python(pending_event)
    except ValidationError:
        return {"error": "지원하지 않거나 형식이 올바르지 않은 이벤트입니다."}

    session_id = _state_get(state, "session_id")
    if event.session_id != session_id:
        return {"error": "현재 면접 세션과 일치하지 않는 이벤트입니다."}

    if isinstance(event, AnswerSubmitted) and _state_get(state, "mode") == "voice":
        command = detect_voice_command(event.text)
        common_fields = {
            "session_id": event.session_id,
            "event_id": event.event_id,
            "occurred_at": event.occurred_at,
        }
        if command == "end":
            event = EndRequested(**common_fields)
        elif command == "replay":
            event = ReplayRequested(question_id=event.question_id, **common_fields)

    if isinstance(event, AnswerSubmitted):
        current_question = _state_get(state, "current_question")
        if current_question is None:
            return {"error": "답변을 연결할 현재 질문이 없습니다."}

        if event.question_id != current_question.question_id:
            return {"error": "현재 질문과 일치하지 않는 답변입니다."}

        if not event.text.strip():
            return {"error": "답변 내용을 입력해 주세요."}

    delivery_metrics = _validated_delivery_metrics(state, event)
    return {
        "pending_event": event.model_dump(mode="json"),
        "pending_delivery_metrics": delivery_metrics,
        "error": None,
    }


def _validated_delivery_metrics(
    state: SessionState | dict[str, Any],
    event: InterviewerEvent,
) -> dict[str, Any] | None:
    """현재 이벤트에서 Assessment로 전달할 수 있는 음성 지표를 정규화한다.

    전달 지표는 음성 모드의 AnswerSubmitted에만 의미가 있다. 다른 모드나
    이벤트에 지표가 포함되면 조용히 제거해 이벤트 계약과 전달 품질 데이터가
    섞이지 않도록 한다. 지표는 답변 내용보다 부가적인 정보이므로 형식이 잘못된
    경우 답변 전체를 실패시키지 않고 None으로 낮춘다.

    Args:
        state:
            세션 mode와 pending_delivery_metrics를 가진 현재 상태.

        event:
            validate_event가 복원하고 음성 명령 해석까지 마친 이벤트.

    Returns:
        JSON 직렬화 가능한 DeliveryMetrics dict. 음성 답변 지표가 없거나 사용할
        수 없으면 None.
    """
    if _state_get(state, "mode") != "voice" or not isinstance(event, AnswerSubmitted):
        return None

    raw_metrics = _state_get(state, "pending_delivery_metrics")
    if raw_metrics is None:
        return None

    try:
        metrics = (
            raw_metrics
            if isinstance(raw_metrics, DeliveryMetrics)
            else DeliveryMetrics.model_validate(raw_metrics)
        )
    except ValidationError:
        return None
    normalized_metrics = metrics.model_dump(mode="json", exclude_none=True)
    return normalized_metrics or None


def record_candidate_answer(
    state: SessionState | dict[str, Any],
) -> dict[str, Any]:
    """검증된 지원자 답변을 transcript에 기록한다.

    validate_event를 통과한 AnswerSubmitted를 candidate Turn으로 변환한다.
    기존 transcript를 직접 변경하지 않고 새 리스트를 반환하여 LangGraph의
    부분 상태 병합 과정에서 이전 대화 기록이 명확하게 보존되도록 한다.

    이 노드는 답변 평가 전에 실행된다. 따라서 이후 Assessment 평가, 답변
    인용, 모순 검출, 최종 리포트와 대화 화면이 같은 답변 기록을 사용할 수
    있다.

    Args:
        state:
            검증된 pending_event, current_question, transcript를 가진 세션 상태.

    Returns:
        지원자 Turn이 추가된 transcript를 담은 부분 상태. 선행 검증 결과가
        없거나 답변 이벤트가 아니면 error를 담은 부분 상태.
    """
    pending_event = _state_get(state, "pending_event")
    if pending_event is None:
        return {"error": "기록할 답변 이벤트가 없습니다."}

    try:
        event = _EVENT_ADAPTER.validate_python(pending_event)
    except ValidationError:
        return {"error": "기록할 답변 이벤트의 형식이 올바르지 않습니다."}

    if not isinstance(event, AnswerSubmitted):
        return {"error": "답변 제출 이벤트만 지원자 발화로 기록할 수 있습니다."}

    current_question = _state_get(state, "current_question")
    if current_question is None:
        return {"error": "답변을 연결할 현재 질문이 없습니다."}

    candidate_turn = Turn(
        role="candidate",
        text=event.text,
        question_id=event.question_id,
        kind=current_question.kind.value,
    )
    transcript = _state_get(state, "transcript", [])

    return {
        "transcript": [*transcript, candidate_turn],
        "silence_count": 0,
        "silence_action": None,
        "timeout_action": None,
        "error": None,
    }


def evaluate_answer(state: SessionState, runtime: Any) -> dict[str, Any]:
    """현재 답변 내용과 선택적인 음성 전달 지표를 Assessment에 전달한다.

    이 노드는 validate_event와 record_candidate_answer를 통과한
    AnswerSubmitted 경로에서만 실행된다. pending_delivery_metrics는 검증 노드가
    음성 답변에 대해서만 남긴 값이며, Assessment 호출에서 한 번 사용한 뒤
    상태에서 즉시 제거한다. 전달 지표는 이벤트 모델 안에 다시 합치지 않는다.

    Args:
        state:
            현재 질문, 검증된 답변 이벤트와 선택적인 음성 전달 지표를 가진
            세션 상태.

        runtime:
            AssessmentPort가 담긴 InterviewDeps를 제공하는 LangGraph runtime.

    Returns:
        직렬화된 평가 신호와 소비 완료된 전달 지표 상태를 담은 부분 상태.
    """
    deps = _runtime_deps(runtime)
    current_question = _state_get(state, "current_question")
    pending_event = _state_get(state, "pending_event") or {}

    if current_question is None:
        return {
            "pending_delivery_metrics": None,
            "error": "current_question is missing",
            "finished": True,
        }

    signal = deps.assessment.evaluate(

        question=current_question,
        answer_text=pending_event["text"],
        delivery_metrics=_state_get(state, "pending_delivery_metrics"),
    )

    return {
        "last_signal": _serialize_signal(signal),
        "pending_delivery_metrics": None,
        "error": None,
    }


def handle_off_topic(
    state: SessionState | dict[str, Any],
) -> dict[str, Any]:
    """관련 없는 답변 이후 현재 질문을 다시 제시하도록 준비한다."""

    return {
        "turn_type": "off_topic",
        "pending_event": None,
        "pending_delivery_metrics": None,
        "error": None,
    }


def ask_follow_up(state: SessionState | dict[str, Any], runtime: Any) -> dict[str, Any]:
    """평가 신호의 추가 확인 대상을 바탕으로 꼬리 질문을 생성한다.

    Args:
        state:
            현재 질문, 마지막 평가 신호, 파생 질문 횟수를 가진 세션 상태.

        runtime:
            StrategyPort가 담긴 InterviewDeps를 제공하는 LangGraph runtime.

    Returns:
        생성된 꼬리 질문과 증가한 파생 질문 횟수를 담은 부분 상태. 질문 생성에
        필요한 상태가 없거나 부모 질문 연결이 잘못되면 error를 반환한다.
    """
    current_question = _state_get(state, "current_question")
    last_signal = _restore_signal(_state_get(state, "last_signal"))
    if current_question is None or last_signal is None:
        return {"error": "꼬리 질문 생성에 필요한 질문 또는 평가 신호가 없습니다."}

    deps = _runtime_deps(runtime)
    question = deps.strategy.next_follow_up(
        topic=current_question.topic,
        parent_question_id=current_question.question_id,
        target=last_signal.next_probe_target,
    )
    if question.parent_question_id != current_question.question_id:
        return {"error": "생성된 꼬리 질문의 부모 질문 ID가 올바르지 않습니다."}

    return {
        "current_question": question,
        "derived_turn_count": _state_get(state, "derived_turn_count", 0) + 1,
        "pending_event": None,
        "pending_delivery_metrics": None,
        "turn_type": "question",
        "error": None,
    }


def ask_challenge(state: SessionState | dict[str, Any], runtime: Any) -> dict[str, Any]:
    """평가에서 발견한 오개념이나 논리적 허점을 확인할 압박 질문을 생성한다.

    Args:
        state:
            현재 질문, 마지막 평가 신호, 질문 세트 상태를 가진 세션 상태.

        runtime:
            StrategyPort가 담긴 InterviewDeps를 제공하는 LangGraph runtime.

    Returns:
        생성된 압박 질문, 증가한 파생 질문 횟수, challenge 사용 상태를 담은
        부분 상태. 필요한 상태가 없거나 부모 연결이 잘못되면 error를 반환한다.
    """
    current_question = _state_get(state, "current_question")
    last_signal = _restore_signal(_state_get(state, "last_signal"))
    if current_question is None or last_signal is None:
        return {"error": "압박 질문 생성에 필요한 질문 또는 평가 신호가 없습니다."}

    deps = _runtime_deps(runtime)
    question = deps.strategy.next_challenge(
        topic=current_question.topic,
        parent_question_id=current_question.question_id,
        target=last_signal.next_probe_target,
    )
    if question.parent_question_id != current_question.question_id:
        return {"error": "생성된 압박 질문의 부모 질문 ID가 올바르지 않습니다."}

    return {
        "current_question": question,
        "derived_turn_count": _state_get(state, "derived_turn_count", 0) + 1,
        "challenge_used_in_set": True,
        "pending_event": None,
        "pending_delivery_metrics": None,
        "turn_type": "question",
        "error": None,
    }


def ask_confirm_positive(
    state: SessionState | dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """대체로 맞는 답변의 범위나 사실관계를 확인할 질문을 생성한다.

    Args:
        state:
            현재 질문, 마지막 평가 신호, 파생 질문 횟수를 가진 세션 상태.

        runtime:
            StrategyPort가 담긴 InterviewDeps를 제공하는 LangGraph runtime.

    Returns:
        생성된 긍정 확인 질문과 증가한 파생 질문 횟수를 담은 부분 상태. 필요한
        상태가 없거나 부모 질문 연결이 잘못되면 error를 반환한다.
    """
    current_question = _state_get(state, "current_question")
    last_signal = _restore_signal(_state_get(state, "last_signal"))
    if current_question is None or last_signal is None:
        return {"error": "긍정 확인 질문 생성에 필요한 질문 또는 평가 신호가 없습니다."}

    deps = _runtime_deps(runtime)
    question = deps.strategy.next_confirm_positive(
        topic=current_question.topic,
        parent_question_id=current_question.question_id,
        target=last_signal.next_probe_target,
    )
    if question.parent_question_id != current_question.question_id:
        return {"error": "생성된 긍정 확인 질문의 부모 질문 ID가 올바르지 않습니다."}

    return {
        "current_question": question,
        "derived_turn_count": _state_get(state, "derived_turn_count", 0) + 1,
        "pending_event": None,
        "pending_delivery_metrics": None,
        "turn_type": "question",
        "error": None,
    }


def ask_confirm_negative(
    state: SessionState | dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """근거나 이전 답변과 충돌하는 내용을 재확인할 질문을 생성한다.

    Args:
        state:
            현재 질문, 마지막 평가 신호, 파생 질문 횟수를 가진 세션 상태.

        runtime:
            StrategyPort가 담긴 InterviewDeps를 제공하는 LangGraph runtime.

    Returns:
        생성된 부정 확인 질문과 증가한 파생 질문 횟수를 담은 부분 상태. 필요한
        상태가 없거나 부모 질문 연결이 잘못되면 error를 반환한다.
    """
    current_question = _state_get(state, "current_question")
    last_signal = _restore_signal(_state_get(state, "last_signal"))
    if current_question is None or last_signal is None:
        return {"error": "부정 확인 질문 생성에 필요한 질문 또는 평가 신호가 없습니다."}

    deps = _runtime_deps(runtime)
    question = deps.strategy.next_confirm_negative(
        topic=current_question.topic,
        parent_question_id=current_question.question_id,
        target=last_signal.next_probe_target,
    )
    if question.parent_question_id != current_question.question_id:
        return {"error": "생성된 부정 확인 질문의 부모 질문 ID가 올바르지 않습니다."}

    return {
        "current_question": question,
        "derived_turn_count": _state_get(state, "derived_turn_count", 0) + 1,
        "pending_event": None,
        "pending_delivery_metrics": None,
        "turn_type": "question",
        "error": None,
    }


def ask_trap(state: SessionState | dict[str, Any], runtime: Any) -> dict[str, Any]:
    """혼동하기 쉬운 개념을 정확히 구분하는지 확인할 함정 질문을 생성한다.

    Args:
        state:
            현재 질문, 마지막 평가 신호, 파생 질문 횟수를 가진 세션 상태.

        runtime:
            StrategyPort가 담긴 InterviewDeps를 제공하는 LangGraph runtime.

    Returns:
        생성된 함정 질문과 증가한 파생 질문 횟수를 담은 부분 상태. 필요한
        상태가 없거나 부모 질문 연결이 잘못되면 error를 반환한다.
    """
    current_question = _state_get(state, "current_question")
    last_signal = _restore_signal(_state_get(state, "last_signal"))
    if current_question is None or last_signal is None:
        return {"error": "함정 질문 생성에 필요한 질문 또는 평가 신호가 없습니다."}

    deps = _runtime_deps(runtime)
    question = deps.strategy.next_trap(
        topic=current_question.topic,
        parent_question_id=current_question.question_id,
        target=last_signal.next_probe_target,
    )
    if question.parent_question_id != current_question.question_id:
        return {"error": "생성된 함정 질문의 부모 질문 ID가 올바르지 않습니다."}

    return {
        "current_question": question,
        "derived_turn_count": _state_get(state, "derived_turn_count", 0) + 1,
        "pending_event": None,
        "pending_delivery_metrics": None,
        "turn_type": "question",
        "error": None,
    }


def complete_set(state: SessionState | dict[str, Any], runtime: Any) -> dict[str, Any]:
    """현재 메인 질문과 파생 질문으로 구성된 평가 단위를 완료한다.

    Assessment에 현재 질문 세트의 완료를 알린 뒤 세트 단위로 사용하는 제한
    상태를 초기화한다. 다음 메인 질문 생성 또는 최종 리포트 이동 여부는 이
    노드가 상태를 변경한 후 after_complete_set 라우터가 결정한다.

    Args:
        state:
            현재 질문 세트의 메인 질문 ID와 진행 상태를 가진 세션 상태.

        runtime:
            AssessmentPort가 담긴 InterviewDeps를 제공하는 LangGraph runtime.

    Returns:
        challenge, 파생 질문 수, 침묵 횟수가 초기화된 부분 상태. 기준 메인
        질문 ID를 찾을 수 없으면 error를 담은 부분 상태.
    """
    current_question = _state_get(state, "current_question")
    main_question_id = _state_get(state, "main_question_id")
    if main_question_id is None and current_question is not None:
        main_question_id = current_question.question_id

    if main_question_id is None:
        return {"error": "완료할 메인 질문 세트를 찾을 수 없습니다."}

    deps = _runtime_deps(runtime)
    deps.assessment.complete_question_set(
        main_question_id=main_question_id,
    )

    return {
        "challenge_used_in_set": False,
        "derived_turn_count": 0,
        "silence_count": 0,
        "silence_action": None,
        "timeout_action": None,
        "pending_event": None,
        "pending_delivery_metrics": None,
        "error": None,
    }


def check_rubric_eligibility(
    state: SessionState | dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """Find novel shareable questions before running the final-report LLM."""
    return {
        "rubric_sources": [],
        "rubric_share_status": "not_available",
        "rubric_share_approved": False,
        "error": None,
    }

def final_report(state: SessionState | dict[str, Any], runtime: Any) -> dict[str, Any]:
    """Assessment가 만든 최종 평가 리포트를 세션 상태에 저장한다.

    리포트는 LangGraph checkpointer가 안전하게 저장할 수 있도록 Pydantic 모델
    객체가 아닌 JSON 호환 dict로 변환한다. 세션 종료 상태 변경은 다음
    finalize 노드가 담당한다.

    Args:
        state:
            최종 리포트를 생성할 면접 세션 상태.

        runtime:
            AssessmentPort가 담긴 InterviewDeps를 제공하는 LangGraph runtime.

    Returns:
        직렬화된 최종 리포트와 초기화된 error를 담은 부분 상태.
    """
    deps = _runtime_deps(runtime)
    generate_report = getattr(
        deps.assessment,
        "finalize_with_rubrics",
        None,
    )
    approved = bool(_state_get(state, "rubric_share_approved", False))
    raw_sources = _state_get(state, "rubric_sources", []) if approved else []
    rubric_sources = [
        source
        if isinstance(source, RubricSource)
        else RubricSource.model_validate(source)
        for source in raw_sources
    ]
    if deps.rubric_sharing_enabled and callable(generate_report):
        result = generate_report(rubric_sources=rubric_sources)
        if not isinstance(result, ReportGenerationResult):
            result = ReportGenerationResult.model_validate(result)
        report = result.report
        rubric_candidates = result.rubric_candidates
    else:
        report = deps.assessment.finalize()
        rubric_candidates = []
    return {
        "report": report.model_dump(mode="json"),
        "rubric_candidates": rubric_candidates,
        "error": None,
    }


def request_rubric_consent(
    state: SessionState | dict[str, Any],
) -> dict[str, Any]:
    """Ask for sharing before any rubric-generation LLM work is performed."""
    rubric_sources = _state_get(state, "rubric_sources", [])
    payload = interrupt(
        {
            "waiting_for": "rubric_share_consent",
            "message": "내 답변을 공용 평가 기준에 반영해도 되나요?",
            "candidate_count": len(rubric_sources),
        }
    )

    approved = isinstance(payload, dict) and payload.get("share") is True
    return {
        "rubric_share_approved": approved,
        "rubric_share_status": "pending" if approved else "discarded",
        "error": None,
    }


def save_rubric_candidates(
    state: SessionState | dict[str, Any],
) -> dict[str, Any]:
    """Persist report-generated rubrics only after explicit approval."""
    if not _state_get(state, "rubric_share_approved", False):
        return {"error": None}

    candidates = [
        candidate
        if isinstance(candidate, RubricCandidate)
        else RubricCandidate.model_validate(candidate)
        for candidate in _state_get(state, "rubric_candidates", [])
    ]
    if not candidates:
        return {"rubric_share_status": "failed", "error": None}

    try:
        store = get_rubric_store()
        for candidate in candidates:
            store.add_candidate(candidate)
    except Exception as exc:
        _logger.exception(
            "[RUBRIC][SAVE_FAILED] candidate_count=%s error=%s",
            len(candidates),
            exc,
        )
        return {"rubric_share_status": "failed", "error": None}

    return {"rubric_share_status": "shared", "error": None}


def finalize(state: SessionState | dict[str, Any]) -> dict[str, Any]:
    """최종 리포트 생성이 끝난 세션을 종료 상태로 전환한다.

    Args:
        state:
            종료할 현재 면접 세션 상태.

    Returns:
        종료 여부, closing 턴, 정리된 pending 입력을 담은 부분 상태.
    """
    return {
        "finished": True,
        "turn_type": "closing",
        "pending_event": None,
        "pending_delivery_metrics": None,
        "error": None,
    }


def handle_replay(state: SessionState | dict[str, Any]) -> dict[str, Any]:
    """현재 질문을 유지한 채 다시 제시할 수 있도록 입력 상태를 정리한다.

    Args:
        state:
            현재 질문과 처리한 이벤트를 가진 세션 상태.

    Returns:
        replay 턴과 정리된 pending 입력을 담은 부분 상태.
    """
    return {
        "turn_type": "replay",
        "pending_event": None,
        "pending_delivery_metrics": None,
        "error": None,
    }


def handle_silence(
    state: SessionState | dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """침묵 지속 시간과 누적 횟수에 따라 다음 대응을 준비한다.

    임계값보다 짧은 침묵은 횟수에 포함하지 않고 입력 대기로 돌아간다. 유효한
    침묵은 횟수를 증가시킨 뒤 SilencePolicy의 첫 번째/두 번째 행동을 적용한다.
    기본 정책에서는 첫 번째 침묵에 Strategy 힌트를 만들고, 두 번째 침묵에는
    현재 질문을 다시 제시한다. 증가한 횟수가 최대 허용값에 도달하면 개별
    행동보다 타임아웃을 우선한다.

    Args:
        state:
            검증된 SilenceDetected 이벤트, 현재 질문, 침묵 정책과 누적 횟수를
            가진 세션 상태.

        runtime:
            힌트 생성에 사용할 StrategyPort가 담긴 LangGraph runtime.

    Returns:
        침묵 횟수, 다음 침묵 행동, 필요한 경우 생성된 힌트 질문과 정리된
        pending 입력을 담은 부분 상태.
    """
    pending_event = _state_get(state, "pending_event")
    try:
        event = _EVENT_ADAPTER.validate_python(pending_event)
    except ValidationError:
        return {
            "silence_action": "wait",
            "pending_event": None,
            "pending_delivery_metrics": None,
            "error": "침묵 이벤트의 형식이 올바르지 않습니다.",
        }

    if not isinstance(event, SilenceDetected):
        return {
            "silence_action": "wait",
            "pending_event": None,
            "pending_delivery_metrics": None,
            "error": "침묵 감지 이벤트만 침묵 정책으로 처리할 수 있습니다.",
        }

    raw_policy = _state_get(state, "silence_policy")
    policy = (
        raw_policy
        if isinstance(raw_policy, SilencePolicy)
        else SilencePolicy.model_validate(raw_policy or {})
    )
    if event.silence_duration_seconds < policy.hint_threshold_seconds:
        return {
            "silence_action": "wait",
            "pending_event": None,
            "pending_delivery_metrics": None,
            "error": None,
        }

    previous_count = _state_get(state, "silence_count", 0)
    silence_count = previous_count + 1
    if silence_count >= policy.max_events_before_timeout:
        return {
            "silence_count": silence_count,
            "silence_action": "timeout",
            "pending_event": None,
            "pending_delivery_metrics": None,
            "error": None,
        }

    policy_action = policy.first_action if previous_count == 0 else policy.second_action
    if policy_action == "represent":
        return {
            "silence_count": silence_count,
            "silence_action": "replay",
            "turn_type": "replay",
            "pending_event": None,
            "pending_delivery_metrics": None,
            "error": None,
        }

    hint_question = _make_silence_hint(state, runtime)
    return {
        "current_question": hint_question,
        "silence_count": silence_count,
        "silence_action": "hint",
        "turn_type": "hint",
        "pending_event": None,
        "pending_delivery_metrics": None,
        "error": None,
    }


def _make_silence_hint(
    state: SessionState | dict[str, Any],
    runtime: Any,
) -> Question:
    """현재 질문과 마지막 평가 신호를 사용해 침묵 힌트를 생성한다.

    Strategy의 힌트 생성 계약을 한 곳에서 호출해, 계약이나 폴백 방식이 바뀔
    경우 handle_silence의 정책 판단을 수정하지 않고 이 helper만 교체할 수
    있게 한다. 완전한 침묵에는 답변 일부가 없으므로 answer_excerpt는 None을
    전달한다.

    Args:
        state:
            현재 질문과 선택적인 마지막 평가 신호를 가진 세션 상태.

        runtime:
            StrategyPort가 담긴 LangGraph runtime.

    Returns:
        Strategy가 생성한 힌트 Question.

    Raises:
        ValueError:
            힌트를 연결할 현재 질문이 없는 경우.
    """
    current_question = _state_get(state, "current_question")
    if current_question is None:
        raise ValueError("힌트를 연결할 현재 질문이 없습니다.")

    target = None
    try:
        last_signal = _restore_signal(_state_get(state, "last_signal"))
    except ValidationError:
        last_signal = None
    if last_signal is not None:
        target = last_signal.next_probe_target

    deps = _runtime_deps(runtime)
    return deps.strategy.next_hint(
        question=current_question,
        target=target,
        answer_excerpt=None,
    )


def handle_timeout(state: SessionState | dict[str, Any]) -> dict[str, Any]:
    """무응답 타임아웃을 세션 정책에 따라 일시 정지 또는 종료로 변환한다.

    TimeoutPolicy.action이 pause이면 세션을 종료하지 않고 pause_prompt 발화를
    준비한다. end이면 최종 리포트 생성으로 이동할 수 있도록 종료 행동을
    기록한다. 직접 전달된 NoResponseTimeout과 누적 침묵에서 승격된 타임아웃이
    같은 정책을 사용하도록 pending 이벤트 종류에는 의존하지 않는다.

    Args:
        state:
            타임아웃 정책과 현재 세션 종료 상태를 가진 세션 상태.

    Returns:
        timeout 행동, 해당 행동의 발화 종류, 정리된 pending 입력을 담은 부분
        상태. pause에서는 finished를 False로 유지한다.
    """
    raw_policy = _state_get(state, "timeout_policy")
    policy = (
        raw_policy
        if isinstance(raw_policy, TimeoutPolicy)
        else TimeoutPolicy.model_validate(raw_policy or {})
    )

    if policy.action == "pause":
        return {
            "timeout_action": "pause",
            "turn_type": "pause_prompt",
            "finished": False,
            "pending_event": None,
            "pending_delivery_metrics": None,
            "error": None,
        }

    return {
        "timeout_action": "end",
        "pending_event": None,
        "pending_delivery_metrics": None,
        "error": None,
    }


def ask_main(state: SessionState, runtime: Any) -> dict[str, Any]:
    """답변 품질 분기 없이 다음 메인 질문을 생성한다.

    3-3 단계에서는 BONUS/MISCONCEPTION 같은 신호를 의도적으로 무시하고
    항상 다음 메인 질문으로 이동한다.
    """
    deps = _runtime_deps(runtime)
    asked_count = _state_get(state, "asked_count", 0)
    last_signal = _restore_signal(_state_get(state, "last_signal"))
    question = deps.strategy.next_question(last_signal=last_signal)

    return {
        "current_question": question,
        "asked_count": asked_count + 1,
        "main_question_id": question.question_id,
        "main_topic": question.topic,
        "challenge_used_in_set": False,
        "turn_type": "question",
        "pending_event": None,
        "pending_delivery_metrics": None,
    }
