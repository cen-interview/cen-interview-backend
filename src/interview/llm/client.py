"""공용 LLM 클라이언트.

모델 이름·재시도·토큰 설정을 한 곳에서 관리한다. Strategy/Interviewer/Assessment 가
각자 ChatAnthropic 을 만들지 않고 여기서 받아 쓴다 (설정 분산 방지).
"""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from interview.config import settings


@lru_cache
def get_llm(temperature: float = 0.3) -> ChatAnthropic:
    """공용 LLM 인스턴스.

    Args:
        temperature: 질문 생성은 약간 높게, 평가(judge)는 낮게 주는 식으로
            호출부에서 조절한다.
    """
    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=temperature,
        max_retries=3,
    )
