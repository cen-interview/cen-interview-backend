import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from interview.assessment import AssessmentAgent
from interview.interviewer.agent import InterviewerAgent
from interview.interviewer.contracts import AssessmentPort, StrategyPort
from interview.interviewer.session import SessionState, Turn
from interview.schemas.events import AnswerSubmitted, InterviewerEvent, Mode
from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Question
from interview.schemas.report import FinalReport
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal
from interview.strategy import StrategyAgent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import TypeAdapter, ValidationError


_EVENT_ADAPTER = TypeAdapter(InterviewerEvent)


@dataclass
class InterviewDeps:
    """세션 흐름에 필요한 런타임 의존성.

    Strategy/Assessment 같은 에이전트 인스턴스는 직렬화 대상 상태가 아니므로
    SessionState에 넣지 않는다. 현재는 InterviewSession이 직접 보관하고,
    이후 LangGraph 전환 시에는 runtime context로 그대로 옮길 수 있다.

    Attributes:
        strategy:
            다음 질문을 결정하고 생성하는 StrategyAgent.

        assessment:
            답변을 평가하고 최종 리포트를 만드는 AssessmentAgent.

        llm:
            이후 발화 레이어나 자연어 합성 단계에서 사용할 선택적 LLM client.
            지금 단계에서는 None이어도 전체 세션 흐름이 동작해야 한다.
    """

    strategy: StrategyPort
    assessment: AssessmentPort
    llm: object | None = None


def _state_get(state: SessionState | dict[str, Any], key: str, default: Any = None) -> Any:
    """LangGraph 노드/라우터에서 state 값을 안전하게 읽는다.

    왜 필요한가:
        이 파일의 노드 함수들은 `SessionState`를 기준으로 작성되어 있지만,
        LangGraph가 노드나 조건부 라우터에 넘겨주는 state는 상황에 따라
        Pydantic 모델처럼 보일 수도 있고, dict처럼 보일 수도 있다.

        예를 들어 우리가 직접 호출하는 코드에서는 `state.current_question`
        처럼 속성 접근이 자연스럽지만, 그래프 내부 병합/라우팅 과정에서는
        `state["current_question"]` 또는 `state.get("current_question")`
        형태가 필요할 수 있다.

    어떤 용도로 쓰나:
        노드 함수가 state의 실제 형태를 신경 쓰지 않고 값을 읽도록 만드는
        작은 어댑터다.

        - `evaluate_answer()`에서 current_question, pending_event를 읽을 때
        - `ask_main()`에서 asked_count, last_signal을 읽을 때
        - `after_ask()`에서 finished, asked_count, max_questions를 읽을 때

    Args:
        state:
            `SessionState` 객체이거나 같은 필드를 가진 dict.

        key:
            읽고 싶은 state 필드 이름.

        default:
            해당 필드가 없을 때 반환할 기본값.

    Returns:
        state에서 꺼낸 값. 필드가 없으면 default를 반환한다.
    """
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _serialize_signal(signal: AnswerQualitySignal) -> dict[str, Any]:
    """AnswerQualitySignal을 SessionState.last_signal에 저장 가능한 dict로 바꾼다.

    왜 필요한가:
        `AssessmentAgent.evaluate()`는 `AnswerQualitySignal` Pydantic 모델을
        반환한다. 그런데 `SessionState.last_signal` 필드는 현재
        `dict | None`으로 정의되어 있다.

        즉, 평가 결과를 state에 그대로 넣으면 타입이 맞지 않는다. 특히
        LangGraph checkpointer를 붙이는 단계에서는 state가 직렬화되어야 하므로,
        Pydantic 모델 객체보다는 JSON 친화적인 dict가 더 안전하다.

    어떤 용도로 쓰나:
        `evaluate_answer()` 노드가 답변 평가를 끝낸 뒤, 평가 신호를
        `last_signal`에 저장하기 직전에 호출한다.

        저장 형태 예:
            {
                "answer_id": "...",
                "question_id": "...",
                "quality": "sufficient",
                "rationale": [...],
                "next_probe_target": null,
            }

    Args:
        signal:
            Assessment가 반환한 답변 평가 신호.

    Returns:
        JSON으로 직렬화 가능한 dict 형태의 평가 신호.
    """
    return signal.model_dump(mode="json")


def _restore_signal(raw_signal: Any) -> AnswerQualitySignal | None:
    """state에 dict로 저장된 last_signal을 AnswerQualitySignal로 복원한다.

    왜 필요한가:
        `_serialize_signal()` 때문에 `SessionState.last_signal`에는 dict가
        저장된다. 하지만 `StrategyAgent.next_question(last_signal=...)`은
        `AnswerQualitySignal | None`을 받도록 설계되어 있다.

        그래서 state에 저장할 때는 dict로 낮추고, Strategy에 넘길 때는
        다시 Pydantic 모델로 올려야 한다.

    어떤 용도로 쓰나:
        `ask_main()` 노드에서 다음 메인 질문을 만들기 전에 호출한다.
        직전 답변의 quality를 Strategy가 난이도 조정에 사용할 수 있게
        `AnswerQualitySignal` 객체로 되돌린다.

    처리하는 입력:
        - None:
            첫 질문처럼 이전 평가가 없을 때 그대로 None 반환.

        - AnswerQualitySignal:
            이미 모델 객체라면 그대로 반환. 테스트나 직접 호출에서 유용하다.

        - dict:
            LangGraph state에 저장된 형태. `model_validate()`로 복원한다.

    Args:
        raw_signal:
            state에서 읽은 last_signal 값.

    Returns:
        복원된 AnswerQualitySignal 또는 None.
    """
    if raw_signal is None:
        return None
    if isinstance(raw_signal, AnswerQualitySignal):
        return raw_signal
    return AnswerQualitySignal.model_validate(raw_signal)


def _runtime_deps(runtime: Any) -> InterviewDeps:
    """LangGraph runtime context에서 InterviewDeps를 꺼낸다.

    왜 필요한가:
        LangGraph에서 그래프 state에는 직렬화 가능한 세션 값만 넣는 것이
        안전하다. 반대로 `StrategyAgent`, `AssessmentAgent`, LLM client 같은
        런타임 객체는 직렬화 대상이 아니므로 state가 아니라 context로
        주입해야 한다.

        이 프로젝트에서는 그 context 스키마를 `InterviewDeps`로 정의했다.

    어떤 용도로 쓰나:
        노드 함수가 Strategy/Assessment에 접근할 때마다 호출한다.

        - `greet()`:
            `deps.strategy.next_question(last_signal=None)` 호출

        - `evaluate_answer()`:
            `deps.assessment.evaluate(...)` 호출

        - `ask_main()`:
            `deps.strategy.next_question(last_signal=...)` 호출

    왜 dict도 처리하나:
        LangGraph 버전이나 호출 방식에 따라 `runtime.context`가 이미
        `InterviewDeps` 인스턴스일 수도 있고, 같은 키를 가진 dict일 수도 있다.
        이 함수는 두 경우를 모두 받아서 노드 안에서는 항상
        `InterviewDeps`처럼 다룰 수 있게 만든다.

    Args:
        runtime:
            LangGraph가 노드 함수에 전달하는 runtime 객체.

    Returns:
        Strategy/Assessment/LLM 의존성을 담은 InterviewDeps.
    """
    context = runtime.context
    if isinstance(context, InterviewDeps):
        return context
    return InterviewDeps(**context)


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
        "turn_type": "question",
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
        AnswerQuality.BONUS_AVAILABLE: "ask_follow_up",
        AnswerQuality.MISCONCEPTION: "ask_challenge",
        AnswerQuality.CONFIRM_POSITIVE: "ask_confirm_positive",
        AnswerQuality.CONFIRM_NEGATIVE: "ask_confirm_negative",
        AnswerQuality.TRAP_AVAILABLE: "ask_trap",
    }
    return routes.get(last_signal.quality, "complete_set")


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


def _build_graph() -> StateGraph:
    """면접 세션의 LangGraph builder를 조립한다.

    왜 필요한가:
        그래프의 노드/엣지 정의와 컴파일 시점을 분리하기 위해 builder 조립을
        함수로 감싼다. 이렇게 해두면 `get_compiled_graph()`가 캐시된 compiled
        graph를 만들 때마다 같은 구조를 명확하게 재사용할 수 있다.

    그래프 흐름:
        START → greet → wait_event → validate_event → route_event

        답변 이벤트는 transcript 기록과 평가를 거친 뒤 route_quality에서
        파생 질문 또는 complete_set으로 분기한다. 질문 세트가 끝나면 다음
        메인 질문을 만들거나 final_report와 finalize를 거쳐 END로 이동한다.

    Returns:
        아직 compile되지 않은 StateGraph builder.
    """
    builder = StateGraph(SessionState, context_schema=InterviewDeps)
    builder.add_node("greet", greet)
    builder.add_node("wait_event", wait_event)
    builder.add_node("validate_event", validate_event)
    builder.add_node("record_candidate_answer", record_candidate_answer)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("ask_main", ask_main)
    builder.add_node("ask_follow_up", ask_follow_up)
    builder.add_node("ask_challenge", ask_challenge)
    builder.add_node("ask_confirm_positive", ask_confirm_positive)
    builder.add_node("ask_confirm_negative", ask_confirm_negative)
    builder.add_node("ask_trap", ask_trap)
    builder.add_node("complete_set", complete_set)
    builder.add_node("final_report", final_report)
    builder.add_node("finalize", finalize)
    builder.add_node("handle_replay", handle_replay)
    builder.add_node("handle_silence", handle_silence)
    builder.add_node("handle_timeout", handle_timeout)

    builder.add_edge(START, "greet")
    builder.add_edge("greet", "wait_event")
    builder.add_edge("wait_event", "validate_event")
    builder.add_conditional_edges(
        "validate_event",
        route_event,
        {
            "record_candidate_answer": "record_candidate_answer",
            "handle_replay": "handle_replay",
            "handle_silence": "handle_silence",
            "handle_timeout": "handle_timeout",
            "final_report": "final_report",
        },
    )
    builder.add_edge("record_candidate_answer", "evaluate_answer")
    builder.add_conditional_edges(
        "evaluate_answer",
        route_quality,
        {
            "complete_set": "complete_set",
            "ask_follow_up": "ask_follow_up",
            "ask_challenge": "ask_challenge",
            "ask_confirm_positive": "ask_confirm_positive",
            "ask_confirm_negative": "ask_confirm_negative",
            "ask_trap": "ask_trap",
        },
    )
    builder.add_conditional_edges(
        "complete_set",
        after_complete_set,
        {
            "final_report": "final_report",
            "ask_main": "ask_main",
        },
    )
    builder.add_edge("ask_main", "wait_event")
    builder.add_edge("ask_follow_up", "wait_event")
    builder.add_edge("ask_challenge", "wait_event")
    builder.add_edge("ask_confirm_positive", "wait_event")
    builder.add_edge("ask_confirm_negative", "wait_event")
    builder.add_edge("ask_trap", "wait_event")
    builder.add_edge("handle_replay", "wait_event")
    builder.add_edge("handle_silence", "wait_event")
    builder.add_edge("handle_timeout", "final_report")
    builder.add_edge("final_report", "finalize")
    builder.add_edge("finalize", END)
    return builder


@lru_cache(maxsize=1)
def get_compiled_graph():
    """체크포인터가 붙은 compiled graph를 모듈 단위로 1회만 생성한다. -- 싱클턴

    왜 필요한가:
        LangGraph의 `interrupt()` 기반 흐름은 checkpointer와 `thread_id`를
        함께 사용해야 재개(resume)가 가능하다. 여기서는 개발/초기 단계에
        적합한 `InMemorySaver`를 붙여 그래프를 컴파일한다.

        compiled graph는 매 요청마다 새로 만들 필요가 없으므로 `lru_cache`로
        프로세스 안에서 한 번만 생성한다. 이후 API 계층은 이 함수를 호출해
        동일한 compiled graph 인스턴스를 재사용하면 된다.

    Returns:
        `InMemorySaver` checkpointer로 컴파일된 LangGraph 실행 객체.
    """
    builder = _build_graph()
    return builder.compile(checkpointer=InMemorySaver())


class InterviewSession:
    """세션 1건에 필요한 에이전트 묶음 + 상태.

    API 계층은 이 클래스의 인터페이스만 알면 된다. 내부가 나중에
    LangGraph로 바뀌더라도 create_session(), get_session(),
    handle_event() 형태는 유지하는 것이 목표다.

    Attributes:
        deps:
            Strategy/Assessment/LLM 같은 런타임 의존성 묶음.

        strategy:
            deps.strategy의 편의 참조. 첫 질문 생성과 다음 질문 생성에 사용한다.

        assessment:
            deps.assessment의 편의 참조. 답변 평가와 리포트 생성에 사용한다.

        state:
            session.py의 SessionState. 현재 질문, 질문 수, 종료 여부,
            이후 그래프 전환용 pending 값들을 보관한다.

        interviewer:
            실제 이벤트 라우팅을 담당하는 InterviewerAgent.
    """

    def __init__(
        self,
        session_id: str,
        mode: Mode,
        coverage: CoverageMap,
        max_questions: int = 10,
        deps: InterviewDeps | None = None,
    ) -> None:
        self.deps = deps or InterviewDeps(
            strategy=StrategyAgent(coverage),
            assessment=AssessmentAgent(),
        )
        self.strategy = self.deps.strategy
        self.assessment = self.deps.assessment
        self.state = SessionState(
            session_id=session_id, mode=mode, max_questions=max_questions
        )
        self.interviewer = InterviewerAgent(self.state, self.strategy, self.assessment)

    def start(self) -> Question:
        """첫 메인 질문을 생성하고 SessionState의 시작 필드를 세팅한다.

        세팅하는 값:
            - current_question: 첫 질문
            - asked_count: 1
            - main_question_id: 첫 메인 질문 ID
            - main_topic: 첫 메인 질문 주제
        """
        question = self.strategy.next_question(last_signal=None)
        self.state.current_question = question
        self.state.asked_count = 1
        self.state.main_question_id = question.question_id
        self.state.main_topic = question.topic
        return question

    def handle_event(self, event: InterviewerEvent) -> Question | None:
        """사용자 이벤트 1건을 처리한다.

        현재 구현에서는 이벤트별 세부 라우팅을 InterviewerAgent에 위임한다.
        반환값은 다음에 사용자에게 제시할 질문이며, 세션이 종료되면 None이다.
        """
        return self.interviewer.handle(event)

    def is_finished(self) -> bool:
        """세션 종료 여부를 반환한다."""
        return self.state.finished

    def finalize(self) -> FinalReport:
        """면접 종료 후 최종 평가서를 생성한다."""
        return self.assessment.finalize()


# [Stub] 세션 레지스트리. 영속화는 8단계에서 실제 DB/Redis 등으로 교체.
_sessions: dict[str, InterviewSession] = {}


def create_session(
    mode: Mode, coverage: CoverageMap | None = None, max_questions: int = 10
) -> tuple[InterviewSession, Question]:
    """세션을 만들고 첫 질문을 반환한다.

    API의 세션 생성 엔드포인트에서 호출하는 진입점이다. 아직 영속 저장소가
    없으므로 생성한 InterviewSession을 모듈 레벨 _sessions에 보관한다.

    Args:
        mode:
            면접 모드. "chat" 또는 "voice".

        coverage:
            Evidence 파이프라인이 만든 주제별 근거 커버리지. 없으면 빈
            CoverageMap을 사용한다.

        max_questions:
            세션에서 물어볼 최대 메인 질문 수.

    Returns:
        생성된 InterviewSession과 첫 질문.
    """
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    session = InterviewSession(session_id, mode, coverage or CoverageMap(), max_questions)
    first_question = session.start()
    _sessions[session_id] = session
    return session, first_question


def get_session(session_id: str) -> InterviewSession:
    """세션 조회. API 의 POST /events 가 호출.

    현재는 인메모리 dict에서 바로 꺼낸다. 서버 재시작, 멀티 프로세스,
    장시간 세션 복구는 아직 지원하지 않는다. 이후 LangGraph checkpointer나
    DB/Redis 기반 세션 저장소로 교체할 수 있다.

    TODO(담당 C): 존재하지 않는 session_id 에 대한 에러 처리(404 매핑)는
    API 계층(5단계)에서.
    """
    return _sessions[session_id]
