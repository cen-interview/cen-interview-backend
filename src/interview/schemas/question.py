"""
Question — Strategy가 생성하여 Interviewer가 사용자에게 제시하는 질문 모델.

Strategy는 답변 평가 신호와 근거 자료를 바탕으로 다음 질문을 생성한다.
Interviewer는 이 Question을 받아 채팅에서는 텍스트로 보여주고,
음성 모드에서는 TTS로 변환해 사용자에게 전달한다.
"""


from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# 질문 종류.
#
# main
#   - 새로운 주제에 대해 처음 묻는 일반 질문
#
# follow_up
#   - 답변은 맞지만 설명이 얕거나 추가로 확인할 요소가 있을 때 묻는 꼬리 질문
#
# challenge
#   - 답변에 오개념이나 논리적 허점이 있을 때 더 깊게 검증하는 압박 질문
#
# confirm_positive
#   - 사용자의 답변이 근거와 대체로 일치할 때, 사실관계나 범위를 부드럽게 확인하는 질문
#
# confirm_negative
#   - 사용자의 답변이 이전 답변 또는 Evidence와 충돌할 때, 불일치를 확인하는 질문
#
# trap
#   - 사용자가 개념을 정확히 구분하는지 확인하기 위해 의도적으로 혼동 가능성이 있는 함정 질문
class QuestionKind(str, Enum):
    MAIN = "main"
    FOLLOW_UP = "follow_up"
    CHALLENGE = "challenge"
    CONFIRM_POSITIVE = "confirm_positive"
    CONFIRM_NEGATIVE = "confirm_negative"
    TRAP = "trap"



# ⚠️ 합의 포인트
# 질문 난이도.
#
# easy
#   - 기본 개념 확인
#
# medium
#   - 개념 간 관계, 사용 이유, 간단한 적용 사례 확인
#
# hard
#   - 실제 프로젝트 적용, 트러블슈팅, 깊이 있는 비교 질문
class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class QuestionCategory(str, Enum):
    TECHNICAL_CONCEPT = "technical_concept"
    PROJECT_IMPLEMENTATION = "project_implementation"
    TROUBLESHOOTING = "troubleshooting"

class Question(BaseModel):
    question_id: str
    text: str
    topic: str
    difficulty: Difficulty

    # 면접 흐름에서의 질문 역할
    kind: QuestionKind = QuestionKind.MAIN

    # 질문 내용의 평가 유형
    category: QuestionCategory

    evidence_ids: list[str] = Field(default_factory=list)
    parent_question_id: str | None = None
