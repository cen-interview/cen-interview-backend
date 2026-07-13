"""LangSmith Studio에서 질문 생성 그래프를 불러오는 개발용 진입점."""

from interview.strategy.graph import get_compiled_graph


# 질문 생성 그래프는 자체 checkpointer를 사용하지 않아 그대로 등록할 수 있다.
graph = get_compiled_graph()
