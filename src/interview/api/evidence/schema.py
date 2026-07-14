"""마이페이지가 소비하는 Evidence 출처 및 인덱싱 API 스키마."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

from interview.schemas.evidence import CoverageMap, IndexBuildResult


EvidenceSourceType = Literal["github", "notion"]


class EvidenceSourceCreateRequest(BaseModel):
    """마이페이지에서 추가할 외부 자료 링크."""

    source_type: EvidenceSourceType
    url: HttpUrl


class EvidenceSourceResponse(BaseModel):
    """등록된 외부 자료 링크의 화면 표시용 정보."""

    id: int
    source_type: EvidenceSourceType
    url: str
    normalized_url: str
    created_at: datetime
    updated_at: datetime


class EvidenceSourceListResponse(BaseModel):
    """현재 사용자가 등록한 Evidence 링크 목록."""

    sources: list[EvidenceSourceResponse] = Field(default_factory=list)


class EvidenceIndexRequest(BaseModel):
    """기존 직접 링크 요청과 저장된 링크 기반 요청을 모두 받는다."""

    notion_links: list[str] = Field(default_factory=list)
    github_links: list[str] = Field(default_factory=list)
    source_ids: list[int] = Field(default_factory=list)


class EvidenceIndexStatus(BaseModel):
    """사용자별 Evidence 인덱싱 진행 상태."""

    status: Literal["idle", "running", "success", "partial_failed", "failed"]
    user_id: int
    started_at: str | None = None
    updated_at: str | None = None
    result: IndexBuildResult | None = None


class EvidenceSummaryResponse(BaseModel):
    """마이페이지 분석 카드에 표시할 사용자별 Evidence 요약."""

    index_status: Literal["idle", "running", "success", "partial_failed", "failed"]
    last_indexed_at: str | None = None
    source_counts: dict[EvidenceSourceType, int]
    raw_doc_count: int = 0
    chunk_count: int = 0
    coverage_map: CoverageMap = Field(default_factory=CoverageMap)
