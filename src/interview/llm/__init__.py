"""공용 LLM 래퍼 (모델 설정/재시도를 한 곳에서 관리)."""

from interview.llm.client import get_llm

__all__ = ["get_llm"]
