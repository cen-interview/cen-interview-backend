"""Interviewer LangGraph에서 상태를 변경하거나 외부 의존성을 호출하는 노드."""

from typing import Any

from interview.interviewer.workflow.runtime import (
    _restore_signal,
    _runtime_deps,
    _serialize_signal,
    _state_get,
)
from interview.interviewer.session import SessionState, Turn
from interview.schemas.events import AnswerSubmitted, InterviewerEvent
from langgraph.types import interrupt
from pydantic import TypeAdapter, ValidationError

_EVENT_ADAPTER = TypeAdapter(InterviewerEvent)

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
    다른 부작용을 두지 않는다.
    """
    payload = interrupt({"waiting_for": "candidate"})

    return {
        "pending_event": payload["event"],
        "pending_delivery_metrics": payload.get("delivery_metrics"),
    }


def validate_event(state: SessionState | dict[str, Any]) -> dict[str, Any]:
    """대기 중인 이벤트를 복원하고 현재 세션에서 처리할 수 있는지 검증한다.

    체크포인터에 dict로 저장된 pending_event를 InterviewerEvent 타입으로
    복원하여 지원하는 이벤트인지 확인한다. 이어서 세션 ID를 검증하고, 답변
    제출 이벤트라면 현재 질문 ID와 빈 답변 여부도 확인한다.

    검증 실패는 예외로 그래프를 중단하지 않고 error에 사용자가 이해할 수 있는
    메시지를 저장한다. 성공한 이벤트도 Pydantic 객체 자체를 상태에 넣지 않고
    JSON 직렬화가 가능한 dict로 다시 변환한다.

    Args:
        state:
            pending_event와 현재 세션 정보를 가진 SessionState 또는 같은 필드를
            가진 dict.

    Returns:
        검증에 성공하면 정규화된 pending_event와 error=None을 담은 부분 상태.
        실패하면 원인을 설명하는 error를 담은 부분 상태.
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

    if isinstance(event, AnswerSubmitted):
        current_question = _state_get(state, "current_question")
        if current_question is None:
            return {"error": "답변을 연결할 현재 질문이 없습니다."}

        if event.question_id != current_question.question_id:
            return {"error": "현재 질문과 일치하지 않는 답변입니다."}

        if not event.text.strip():
            return {"error": "답변 내용을 입력해 주세요."}

    return {
        "pending_event": event.model_dump(mode="json"),
        "error": None,
    }


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
        "error": None,
    }


def evaluate_answer(state: SessionState, runtime: Any) -> dict[str, Any]:
    """현재 질문과 pending_event의 답변 텍스트를 평가한다.

    3단계 skeleton에서는 이벤트 타입 검증과 분기를 아직 하지 않는다.
    따라서 pending_event는 answer_submitted 형태라고 가정한다.
    """
    deps = _runtime_deps(runtime)
    current_question = _state_get(state, "current_question")
    pending_event = _state_get(state, "pending_event") or {}

    if current_question is None:
        return {"error": "current_question is missing", "finished": True}

    signal = deps.assessment.evaluate(

        question=current_question,
        answer_text=pending_event["text"],
        delivery_metrics=_state_get(state, "pending_delivery_metrics"),
    )

    return {
        "last_signal": _serialize_signal(signal),
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
    deps.assessment.complete_question_set(main_question_id=main_question_id)

    return {
        "challenge_used_in_set": False,
        "derived_turn_count": 0,
        "silence_count": 0,
        "pending_event": None,
        "pending_delivery_metrics": None,
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
    report = deps.assessment.finalize()
    return {
        "report": report.model_dump(mode="json"),
        "error": None,
    }


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


def handle_silence(state: SessionState | dict[str, Any]) -> dict[str, Any]:
    """침묵 횟수를 증가시키고 현재 질문을 다시 제시하도록 준비한다.

    상세한 힌트 및 반복 정책은 이후 침묵 처리 단계에서 확장한다.

    Args:
        state:
            현재 침묵 횟수와 질문을 가진 세션 상태.

    Returns:
        증가한 침묵 횟수, replay 턴, 정리된 pending 입력을 담은 부분 상태.
    """
    return {
        "silence_count": _state_get(state, "silence_count", 0) + 1,
        "turn_type": "replay",
        "pending_event": None,
        "pending_delivery_metrics": None,
        "error": None,

    }


def handle_timeout(state: SessionState | dict[str, Any]) -> dict[str, Any]:
    """무응답 타임아웃 이벤트를 종료 준비 상태로 전환한다.

    상세한 end/pause 정책 분기는 이후 타임아웃 처리 단계에서 확장한다.

    Args:
        state:
            타임아웃 이벤트가 검증된 현재 세션 상태.

    Returns:
        closing 턴과 정리된 pending 입력을 담은 부분 상태.
    """
    return {
        "turn_type": "closing",
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
