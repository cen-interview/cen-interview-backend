"""
AnswerQualitySignal — Assessment가 생성하여 Interviewer에게 전달하는 평가 결과.

Interviewer는 quality 값만 보고 다음 면접 흐름을 결정한다.

  sufficient        → Strategy.next_question()          (다음 메인 질문)
  bonus_available   → Strategy.next_follow_up()         (추가 확인 가능한 꼬리 질문)
  misconception     → Strategy.next_challenge()         (오개념/논리 허점에 대한 압박 질문)
  confirm_positive  → Strategy.next_confirm_positive()  (긍정 확인 질문)
  confirm_negative  → Strategy.next_confirm_negative()  (부정 확인 질문)
  trap_available    → Strategy.next_trap()              (함정 질문)

quality는 Assessment와 Interviewer 사이의 핵심 신호이며,
면접 진행 방향을 결정하는 기준이 된다.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# 답변 품질/분기 신호.
#
# sufficient
#   - 핵심 개념을 충분히 설명했고 Evidence와도 일치함
#   - 추가 꼬리질문 없이 다음 메인 질문으로 진행
#
# bonus_available
#   - 답변은 대체로 맞지만, Evidence상 더 깊게 물어볼 만한 요소가 있음
#   - 추가 설명을 끌어내는 꼬리 질문 생성
#
# misconception
#   - 답변에 오개념, 논리적 허점, 과도한 일반화가 포함됨
#   - 사용자의 이해를 더 깊게 검증하는 압박 질문 생성
#
# confirm_positive
#   - 답변이 Evidence와 대체로 일치하지만, 범위나 사실관계를 한 번 더 확인할 필요가 있음
#   - 긍정 확인 질문 생성
#
# confirm_negative
#   - 답변이 Evidence 또는 이전 답변과 충돌함
#   - 불일치 여부를 확인하는 부정 확인 질문 생성
#
# trap_available
#   - 사용자가 헷갈리기 쉬운 개념을 정확히 구분하는지 확인할 필요가 있음
#   - 의도적으로 혼동 가능성이 있는 함정 질문 생성

class AnswerQuality(str, Enum):
    SUFFICIENT = "sufficient"
    BONUS_AVAILABLE = "bonus_available"
    MISCONCEPTION = "misconception"
    CONFIRM_POSITIVE = "confirm_positive"
    CONFIRM_NEGATIVE = "confirm_negative"
    TRAP_AVAILABLE = "trap_available"



class AnswerQualitySignal(BaseModel):
    answer_id: str
    question_id: str
    quality: AnswerQuality

    # 다음 질문에서 무엇을 파고들지
    next_probe_target: str | None = None

    # 해당 quality로 판단한 핵심 평가 요소
    rationale: list[str] = Field(default_factory=list)



    """
    quality
    → 어떤 질문 유형으로 갈지 결정

    next_probe_target
    → 다음 질문에서 뭘 물을지 결정

    rationale
    → 나중에 평가 코멘트 만들 때 참고
    
    """