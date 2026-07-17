"""LangGraph 노드가 공유하는 런타임 의존성과 상태 변환 도구."""

from dataclasses import dataclass
from typing import Any

from interview.interviewer.contracts import AssessmentPort, StrategyPort
from interview.interviewer.session import SessionState
from interview.schemas.signals import AnswerQualitySignal

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
    rubric_sharing_enabled: bool = False


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
