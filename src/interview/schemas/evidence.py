"""
EvidenceChunk + 메타데이터

evidence 인덱싱 파이프라인(면접 전 1회)이 만들어 evidence_store(vector DB)에 적재하고,
런타임에는 search_evidence(query, topic) 가 관련 chunk 를 반환한다.

이 스키마는 Strategy(질문 생성)와 Assessment(답변 평가)가 공통으로 소비한다.
→ 그래서 evidence 담당(A) 혼자 모양을 정하면 안 되고, B/D 와 합의해야 한다.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# 근거 출처 종류. 새 소스(예: 블로그)를 붙일 일이 없으면 이 둘로 충분.
SourceType = Literal["notion", "github"]


class EvidenceChunk(BaseModel):
    """면접 근거 한 조각 (개념 설명 / 코드 조각 / 회고 등)."""
    chunk_id: str
    text: str  # 실제 근거 내용

    # --- 출처 메타데이터 ---
    source_type: SourceType
    source_url: str            # 원본 Notion 페이지 / GitHub 파일 URL
    topic: str                 # 기술 주제 (예: "JPA N+1", "JWT 인증")
    doc_type: Optional[str] = None  # "주차정리"/"회고"/"코드"/"README" 등
    week: Optional[int] = None      # 주차 (Notion 주차 기록일 때)
    date: Optional[str] = None      # 날짜 (ISO 문자열, 예 "2026-03-01")

    # 신뢰도: 내용이 부족한 주제는 낮게 표시한다. (0.0 ~ 1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    """
    search_evidence() 가 반환하는 한 건.
    EvidenceChunk 자체 + '이번 쿼리에서의' 관련도 점수를 함께 준다.
    (점수는 chunk 고유값이 아니라 검색마다 달라지므로 분리)
    """
    chunk: EvidenceChunk
    score: float  # 쿼리와의 관련도 (재랭킹 점수). 높을수록 관련 큼.
