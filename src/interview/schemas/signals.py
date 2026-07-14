"""
AnswerQualitySignal — Assessment가 생성하여 Interviewer에게 전달하는 평가 신호.

Assessment는 사용자의 답변을 Evidence와 비교하여 답변 상태를 판단한다.
Interviewer는 이 신호의 quality 값을 보고 다음 질문 흐름만 결정한다.

즉, 이 스키마는 '답변 평가 상세 결과'라기보다
'다음 면접 흐름을 결정하기 위한 최소 신호'이다.

흐름:
  sufficient        → Strategy.next_question()
  bonus_available   → Strategy.next_follow_up()
  misconception     → Strategy.next_challenge()
  confirm_positive  → Strategy.next_confirm_positive()
  confirm_negative  → Strategy.next_confirm_negative()
  trap_available    → Strategy.next_trap()


class AnswerQualitySignal:
  answer_id
    - 어떤 답변에 대한 평가인지 구분하기 위해 필요하다.
    - 나중에 답변 로그, 평가 기록, 리포트와 연결할 수 있다.

  question_id
    - 이 답변이 어떤 질문에 대한 답변인지 알기 위해 필요하다.
    - 꼬리 질문, 압박 질문, 확인 질문이 어떤 원 질문에서 파생됐는지 추적할 수 있다.

  quality
    - 다음 면접 흐름을 결정하는 핵심 값이다.
    - Interviewer는 이 값만 보고 Strategy의 어떤 함수를 호출할지 결정한다.

  rationale
    - 왜 해당 quality로 판단했는지에 대한 평가 근거이다.
    - Interviewer 라우팅에는 직접 필요하지 않지만, 로그/디버깅/최종 리포트 생성에 활용된다.

  next_probe_target
    - 다음 질문에서 집중적으로 파고들 대상을 나타낸다.
    - 예: "fetch join", "지연 로딩", "트랜잭션 전파"
    - 필수는 아니며, 추가 질문이 필요 없으면 None이 될 수 있다.
    
  unknown
    - 사용자가 모른다고 답했거나 답변 내용이 거의 없어 평가할 수 없음
    - 오답이나 오개념으로 단정하지 않고 현재 질문을 종료
    - 다음 메인 질문으로 이동
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
#
# unknown
#   - 사용자가 개념을 모른다고 답했거나 답변 내용이 없어 평가할 수 없음

class AnswerQuality(str, Enum):
    SUFFICIENT = "sufficient"
    BONUS_AVAILABLE = "bonus_available"
    MISCONCEPTION = "misconception"
    CONFIRM_POSITIVE = "confirm_positive"
    CONFIRM_NEGATIVE = "confirm_negative"
    TRAP_AVAILABLE = "trap_available"
    UNKNOWN = "unknown"


class ConflictType(str, Enum):
    EVIDENCE_CONFLICT = "evidence_conflict"
    SELF_CONTRADICTION = "self_contradiction"


class AnswerQualitySignal(BaseModel):
    answer_id: str
    question_id: str
    quality: AnswerQuality

    rationale: list[str] = Field(default_factory=list)
    conflict_type: ConflictType | None = None
    
    next_probe_target: Optional[str] = None

    accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    sufficiency: float = Field(default=0.0, ge=0.0, le=1.0)

    delivery_note: str | None = None



