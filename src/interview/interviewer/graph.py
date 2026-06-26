"""LangGraph 전체 흐름 배선.

세 에이전트(Strategy/Interviewer/Assessment)와 Evidence 툴을 하나의 그래프로
연결한다. 이 파일이 "접착제"다 — 주인 없으면 아무도 안 짜는 영역이므로
Interviewer 담당(C)이 관리한다.

노드(개략):
  start ─▶ first_question ─▶ wait_event ─▶ route ─▶ (next/follow/hint/confirm)
                                              └─▶ end ─▶ final_report

state 는 interviewer.session.SessionState 를 사용한다.
"""

from langgraph.graph import END, StateGraph

from interview.interviewer.session import SessionState


def build_graph():
    """면접 흐름 그래프를 만들어 컴파일한다.

    TODO(담당 C):
      - StateGraph(SessionState) 로 그래프 구성
      - 노드 등록: pick_first_question / handle_event / make_report ...
      - 조건부 엣지: session.is_done() / quality 신호로 분기
      - checkpointer 로 세션 상태 영속화 (음성 모드 재접속 대비)
    """
    graph = StateGraph(SessionState)
    # graph.add_node("first_question", ...)
    # graph.add_node("handle_event", ...)
    # graph.add_node("final_report", ...)
    # graph.set_entry_point("first_question")
    # graph.add_conditional_edges("handle_event", route_fn, {...})
    # graph.add_edge("final_report", END)
    raise NotImplementedError
