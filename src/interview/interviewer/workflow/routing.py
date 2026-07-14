"""Interviewer LangGraph의 부작용 없는 조건부 라우팅 함수."""

from typing import Any

from interview.interviewer.session import SessionState
from interview.interviewer.workflow.runtime import _restore_signal, _state_get
from interview.schemas.signals import AnswerQuality
from pydantic import ValidationError

def route_event(state: SessionState | dict[str, Any]) -> str:
    """검증 결과와 이벤트 종류를 읽어 다음 처리 노드를 선택한다.

    라우팅 함수는 상태를 변경하거나 Strategy, Assessment 같은 외부
    의존성을 호출하지 않는다. validate_event가 남긴 error가 있으면 현재
    질문을 다시 제시하도록 handle_replay를 선택한다. 유효한 이벤트라면
    pending_event의 type 값만 사용해 이벤트별 처리 노드를 반환한다.

    AnswerSubmitted는 지원자 답변을 transcript에 먼저 기록해야 하므로
    record_candidate_answer로 보낸다. 기록이 끝난 뒤 evaluate_answer로
    연결하는 edge는 그래프 조립 단계에서 정의한다.

    Args:
        state:
            검증된 pending_event와 error를 가진 SessionState 또는 같은 필드를
            가진 dict.

    Returns:
        이벤트를 처리할 다음 그래프 노드 이름. 검증 오류나 알 수 없는 이벤트는
        안전하게 handle_replay로 보낸다.
    """
    if _state_get(state, "error") is not None:
        return "handle_replay"

    pending_event = _state_get(state, "pending_event") or {}
    event_type = pending_event.get("type")

    routes = {
        "answer_submitted": "record_candidate_answer",
        "replay_requested": "handle_replay",
        "silence_detected": "handle_silence",
        "no_response_timeout": "handle_timeout",
        "end_requested": "final_report",
    }
    return routes.get(event_type, "handle_replay")


def route_quality(state: SessionState | dict[str, Any]) -> str:
    """답변 품질과 질문 세트 제한을 읽어 다음 처리 노드를 선택한다.

    파생 질문이 반복되어 면접이 끝나지 않는 것을 방지하기 위해 일반적인
    quality 분기보다 제한 규칙을 먼저 검사한다. 현재 질문 세트의 파생 질문
    수가 최대값에 도달했거나 이미 challenge를 사용한 뒤 다시 misconception이
    나오면 추가 질문을 만들지 않고 complete_set으로 이동한다.

    이 함수는 상태를 읽기만 하며 값을 변경하거나 Strategy와 Assessment를
    호출하지 않는다.

    Args:
        state:
            last_signal과 현재 질문 세트의 제한 상태를 가진 SessionState 또는
            같은 필드를 가진 dict.

    Returns:
        답변 품질에 대응하는 다음 그래프 노드 이름. 평가 신호가 없거나 형식이
        올바르지 않은 경우에는 안전하게 complete_set을 반환한다.
    """
    derived_turn_count = _state_get(state, "derived_turn_count", 0)
    max_derived_turns = _state_get(state, "max_derived_turns_per_set", 2)
    if derived_turn_count >= max_derived_turns:
        return "complete_set"

    try:
        last_signal = _restore_signal(_state_get(state, "last_signal"))
    except ValidationError:
        return "complete_set"

    if last_signal is None:
        return "complete_set"

    if (
        last_signal.quality == AnswerQuality.MISCONCEPTION
        and _state_get(state, "challenge_used_in_set", False)
    ):
        return "complete_set"

    routes = {
        AnswerQuality.SUFFICIENT: "complete_set",
        AnswerQuality.UNKNOWN: "complete_set",
        AnswerQuality.BONUS_AVAILABLE: "ask_follow_up",
        AnswerQuality.MISCONCEPTION: "ask_challenge",
        AnswerQuality.CONFIRM_POSITIVE: "ask_confirm_positive",
        AnswerQuality.CONFIRM_NEGATIVE: "ask_confirm_negative",
        AnswerQuality.TRAP_AVAILABLE: "ask_trap",
    }
    return routes.get(last_signal.quality, "complete_set")


def after_handle_silence(state: SessionState | dict[str, Any]) -> str:
    """침묵 정책의 처리 결과에 따라 다음 그래프 노드를 선택한다.

    짧은 침묵은 새 발화를 만들지 않고 곧바로 입력 대기로 돌아간다. 유효한
    첫 침묵의 힌트와 두 번째 침묵의 질문 재제시는 발화 조립으로 보내고,
    허용 횟수에 도달한 침묵은 타임아웃 처리 노드로 보낸다. 이 함수는 상태를
    읽기만 하며 변경하지 않는다.

    Args:
        state:
            handle_silence가 결정한 silence_action을 가진 세션 상태.

    Returns:
        다음에 실행할 ``wait_event``, ``compose_utterance`` 또는
        ``handle_timeout`` 노드 이름.
    """
    routes = {
        "wait": "wait_event",
        "hint": "compose_utterance",
        "replay": "compose_utterance",
        "timeout": "handle_timeout",
    }
    return routes.get(_state_get(state, "silence_action"), "wait_event")


def after_handle_timeout(state: SessionState | dict[str, Any]) -> str:
    """타임아웃 정책의 처리 결과에 따라 일시 정지 또는 종료 경로를 선택한다.

    pause는 면접을 종료하지 않고 안내 문장을 조립한 뒤 다시 입력을 기다리는
    경로다. end는 Assessment 최종 리포트를 생성하는 경로다. 알 수 없는 값은
    면접 세션이 무기한 남는 것을 방지하기 위해 안전하게 종료 경로로 보낸다.
    이 함수는 상태를 읽기만 하며 변경하지 않는다.

    Args:
        state:
            handle_timeout이 결정한 timeout_action을 가진 세션 상태.

    Returns:
        일시 정지 안내를 만들면 ``compose_utterance``, 종료 처리를 시작하면
        ``final_report`` 노드 이름.
    """
    if _state_get(state, "timeout_action") == "pause":
        return "compose_utterance"
    return "final_report"


def after_complete_set(state: SessionState | dict[str, Any]) -> str:
    """질문 세트 완료 후 다음 메인 질문 또는 최종 리포트 경로를 선택한다.

    asked_count는 지금까지 생성된 메인 질문 수다. 종료 판단을 질문 생성 직후가
    아니라 complete_set 이후에 수행하므로 마지막 메인 질문의 답변과 평가가
    끝난 뒤에만 최종 리포트로 이동한다. 이 함수는 상태를 읽기만 한다.

    Args:
        state:
            메인 질문 수와 세션 종료 여부를 가진 세션 상태.

    Returns:
        최대 질문 수에 도달했거나 종료 상태이면 "final_report", 아니면
        "ask_main".
    """
    if _state_get(state, "finished", False):
        return "final_report"

    asked_count = _state_get(state, "asked_count", 0)
    max_questions = _state_get(state, "max_questions", 10)
    return "final_report" if asked_count >= max_questions else "ask_main"
