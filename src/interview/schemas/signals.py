"""답변 평가 신호.

Assessment 가 답변을 평가한 결과를 Interviewer 에게 넘기는 계약.
Interviewer 는 이 신호의 quality 값만 보고 다음 행동(다음 질문/꼬리/힌트/확인)을
라우팅한다. 설계 문서의 answer_quality_signal / missing_keywords 에 해당.
"""

from enum import Enum

from pydantic import BaseModel, Field


class QualityLevel(str, Enum):
    SUFFICIENT = "sufficient"  # 충분 → 다음 질문
    SHALLOW = "shallow"        # 얕음/애매 → 꼬리 질문
    STUCK = "stuck"            # 막힘 → 힌트성 질문
    CONFLICT = "conflict"      # 이전 답변과 충돌 → 확인 질문


class AnswerQualitySignal(BaseModel):
    """한 답변에 대한 평가 신호 (Interviewer 라우팅용)."""

    quality: QualityLevel
    missing_keywords: list[str] = Field(default_factory=list)
    # CONFLICT 인 경우, 어떤 이전 답변/질문과 충돌하는지
    conflict_with_question_id: str | None = None
    # 사람이 읽을 수 있는 한 줄 근거 (디버깅/로그용)
    rationale: str = ""
