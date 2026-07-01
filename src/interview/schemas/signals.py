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

from typing import Optional

from pydantic import BaseModel, Field
from enum import Enum

# ⚠️ 합의 포인트: 답변 품질 종류.
#   sufficient     = 답변이 충분함 → 다음 일반 질문
#   shallow        = 답변이 얕거나 핵심 개념이 누락됨 → 꼬리 질문
#   misconception  = 오개념이 있음 → 확인 질문

class AnswerQuality(str, Enum):
    SUFFICIENT = "sufficient"
    SHALLOW = "shallow"
    MISCONCEPTION = "misconception"

class AnswerQualitySignal(BaseModel):
    question_id: str
    quality: AnswerQuality

    # 답변에서 빠진/짚은 핵심 키워드.
    # → Strategy 가 꼬리질문/확인 질문을 만들 때 "뭘 더 물을지" 재료로 쓴다.
    #   예: missing_keywords = ["fetch join", "지연 로딩"]
    missing_keywords: list[str] = Field(default_factory=list)
    covered_keywords: list[str] = Field(default_factory=list)

    # quality == AnswerQuality.MISCONCEPTION 일 때:
    # 어떤 오개념이 있는지 설명한다.
    misconception_note: Optional[str] = None

    # 왜 이렇게 판단했는지 (로그/디버깅용, 사용자에겐 안 보여줘도 됨)
    rationale: Optional[str] = None