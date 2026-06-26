"""Retrieval Tool (런타임 공용).

Strategy(질문 생성)와 Assessment(답변 평가)가 각자 필요할 때 호출하는 공용 조회 툴.
"추가 수집"이 아니라 이미 구축된 store 에 대한 검색/재랭킹이다.

LangChain Tool 로도 노출해 에이전트가 tool calling 으로 부를 수 있게 한다.
"""

from langchain_core.tools import tool

from interview.evidence.store import get_store
from interview.schemas.evidence import EvidenceChunk


def search_evidence(
    query: str, topic: str | None = None, k: int = 5
) -> list[EvidenceChunk]:
    """evidence_store 에서 관련 근거 chunk 를 반환한다.

    Strategy / Assessment 가 직접 import 해서 호출하는 일반 함수 버전.
    """
    return get_store().query(query=query, topic=topic, k=k)


@tool
def search_evidence_tool(query: str, topic: str | None = None) -> str:
    """학습 기록/프로젝트 근거에서 query 와 관련된 내용을 찾는다.
    LLM tool calling 용. (topic 으로 주제를 좁힐 수 있음)
    """
    chunks = search_evidence(query=query, topic=topic)
    # LLM 에 넘기기 좋게 텍스트로 직렬화
    return "\n\n".join(f"[{c.source_type}] {c.text}" for c in chunks)
