"""
AnswerQualitySignal — Assessment 가 만들어 Interviewer 에게 넘긴다.

이 신호의 quality 값이 면접 흐름의 '심장'이다.
Interviewer 는 이 값만 보고 다음 행동을 라우팅한다:

  sufficient → Strategy.next_question   (다음 질문)
  shallow    → Strategy.next_follow_up  (꼬리 질문)
  stuck      → Strategy.next_hint       (힌트 질문)
  conflict   → 확인 질문

⚠️ 합의 포인트: quality 4종으로 충분한가? 가장 먼저 합의할 값.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

AnswerQuality = Literal["sufficient", "shallow", "stuck", "conflict"]


class AnswerQualitySignal(BaseModel):
    question_id: str
    quality: AnswerQuality

    # 답변에서 빠진/짚은 핵심 키워드.
    # → Strategy 가 꼬리질문/힌트를 만들 때 "뭘 더 물을지" 재료로 쓴다.
    #   (설계 예시: missing_keywords = ["fetch join", "지연 로딩"])
    missing_keywords: list[str] = Field(default_factory=list)
    covered_keywords: list[str] = Field(default_factory=list)

    # quality == "conflict" 일 때: 어떤 이전 답변과 충돌하는지
    conflict_with_question_id: Optional[str] = None

    # 왜 이렇게 판단했는지 (로그/디버깅용, 사용자에겐 안 보여줘도 됨)
    rationale: Optional[str] = None
