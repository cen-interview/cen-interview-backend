"""질문(Question) 관련 계약.

Strategy 가 생성해서 Interviewer 에게 넘기는 질문의 모양.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionKind(str, Enum):
    MAIN = "main"            # 일반 질문
    FOLLOW_UP = "follow_up"  # 꼬리 질문 (답변이 얕을 때)
    HINT = "hint"            # 힌트성 질문 (막혔을 때)
    CONFIRM = "confirm"      # 확인 질문 (이전 답변과 충돌할 때)


class Question(BaseModel):
    """면접 질문 1건."""

    question_id: str
    text: str
    topic: str
    difficulty: Difficulty
    kind: QuestionKind = QuestionKind.MAIN

    # 이 질문을 만들 때 참고한 근거 chunk_id 들 (추적/평가에 사용)
    linked_evidence: list[str] = Field(default_factory=list)
