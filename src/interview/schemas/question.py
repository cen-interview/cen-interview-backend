"""
Question — Strategy 가 만들어 Interviewer 가 사용자에게 제시한다.
(음성: TTS / 채팅: 텍스트)
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ⚠️ 합의 포인트: 난이도 단계. easy/medium/hard 3단계로 충분한가?
Difficulty = Literal["easy", "medium", "hard"]

# ⚠️ 합의 포인트: 질문 종류.
#   main      = 새 주제 첫 질문
#   follow_up = 꼬리 질문 (답이 얕을 때)
#   hint      = 힌트성 질문 (막혔을 때)
#   confirm   = 확인 질문 (이전 답변과 충돌할 때)
QuestionKind = Literal["main", "follow_up", "hint", "confirm"]


class Question(BaseModel):
    question_id: str
    text: str
    topic: str
    difficulty: Difficulty
    kind: QuestionKind = "main"

    # 이 질문이 어떤 근거에서 나왔는지 (EvidenceChunk.chunk_id 목록)
    # → 나중에 "이 질문 왜 나왔지?" 추적 가능
    evidence_ids: list[str] = Field(default_factory=list)

    # 꼬리/힌트/확인 질문이면, 어떤 질문에서 파생됐는지
    parent_question_id: Optional[str] = None
