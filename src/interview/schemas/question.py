"""
Question — Strategy 가 만들어 Interviewer 가 사용자에게 제시한다.
(음성: TTS / 채팅: 텍스트)
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from enum import Enum

# ⚠️ 합의 포인트: 질문 종류.
#   main      = 새 주제 첫 질문
#   follow_up = 꼬리 질문 (답이 얕을 때)
#   confirm   = 확인 질문 (오개념에 대한 확인)
class QuestionKind(str, Enum):
    MAIN = "main"
    FOLLOW_UP = "follow_up"
    CONFIRM = "confirm"

# ⚠️ 합의 포인트: 난이도 단계. easy/medium/hard 3단계로 충분한가?
class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Question(BaseModel):
    question_id: str
    text: str
    topic: str
    difficulty: Difficulty
    kind: QuestionKind = QuestionKind.MAIN

    # 이 질문이 어떤 근거에서 나왔는지 (EvidenceChunk.chunk_id 목록)
    # → 나중에 "이 질문 왜 나왔지?" 추적 가능
    evidence_ids: list[str] = Field(default_factory=list)

    # 꼬리/확인 질문이면, 어떤 질문에서 파생됐는지
    parent_question_id: Optional[str] = None
