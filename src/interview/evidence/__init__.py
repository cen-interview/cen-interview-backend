"""Evidence: Notion/GitHub 인덱싱 파이프라인 + 런타임 Retrieval Tool.

독립 에이전트가 아니라 "면접 전 1회 구축되는 지식 베이스 + 공용 조회 툴"이다.
  - 면접 전:   build_index() 로 evidence_store 구축
  - 면접 중:   search_evidence() 로 Strategy/Assessment 가 조회
"""

from interview.evidence.indexing import build_index
from interview.evidence.retrieval import search_evidence, search_evidence_tool

__all__ = ["build_index", "search_evidence", "search_evidence_tool"]
