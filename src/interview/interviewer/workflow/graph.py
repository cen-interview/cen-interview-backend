"""Interviewer LangGraph의 노드 등록, 경로 연결, 컴파일 진입점."""

from functools import lru_cache

from interview.interviewer.facade import InterviewSession, create_session, get_session
from interview.interviewer.speech.composition import after_compose_utterance, compose_utterance
from interview.interviewer.workflow.nodes import (
    ask_challenge,
    ask_confirm_negative,
    ask_confirm_positive,
    ask_follow_up,
    ask_main,
    ask_trap,
    complete_set,
    evaluate_answer,
    final_report,
    finalize,
    greet,
    handle_replay,
    handle_silence,
    handle_timeout,
    record_candidate_answer,
    validate_event,
    wait_event,
)
from interview.interviewer.session import SessionState
from interview.interviewer.workflow.routing import after_complete_set, route_event, route_quality
from interview.interviewer.workflow.runtime import InterviewDeps
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

__all__ = [
    "InterviewDeps",
    "InterviewSession",
    "compose_utterance",
    "create_session",
    "get_compiled_graph",
    "get_session",
]


def _build_graph() -> StateGraph:
    """면접 세션의 LangGraph builder를 조립한다.

    왜 필요한가:
        그래프의 노드/엣지 정의와 컴파일 시점을 분리하기 위해 builder 조립을
        함수로 감싼다. 이렇게 해두면 `get_compiled_graph()`가 캐시된 compiled
        graph를 만들 때마다 같은 구조를 명확하게 재사용할 수 있다.

    그래프 흐름:
        START → greet → compose_utterance → wait_event
        → validate_event → route_event

        답변 이벤트는 transcript 기록과 평가를 거친 뒤 route_quality에서
        파생 질문 또는 complete_set으로 분기한다. 질문 세트가 끝나면 다음
        메인 질문을 만들거나 final_report와 finalize를 거쳐 END로 이동한다.

    Returns:
        아직 compile되지 않은 StateGraph builder.
    """
    builder = StateGraph(SessionState, context_schema=InterviewDeps)
    builder.add_node("greet", greet)
    builder.add_node("compose_utterance", compose_utterance)
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
    builder.add_edge("greet", "compose_utterance")
    builder.add_conditional_edges(
        "compose_utterance",
        after_compose_utterance,
        {
            "wait_event": "wait_event",
            "end": END,
        },
    )
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
    builder.add_edge("ask_main", "compose_utterance")
    builder.add_edge("ask_follow_up", "compose_utterance")
    builder.add_edge("ask_challenge", "compose_utterance")
    builder.add_edge("ask_confirm_positive", "compose_utterance")
    builder.add_edge("ask_confirm_negative", "compose_utterance")
    builder.add_edge("ask_trap", "compose_utterance")
    builder.add_edge("handle_replay", "compose_utterance")
    builder.add_edge("handle_silence", "compose_utterance")
    builder.add_edge("handle_timeout", "final_report")
    builder.add_edge("final_report", "finalize")
    builder.add_edge("finalize", "compose_utterance")
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
