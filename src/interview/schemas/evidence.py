"""
EvidenceChunk + 메타데이터

evidence 인덱싱 파이프라인(면접 전 1회)이 만들어 evidence_store(vector DB)에 적재하고,
런타임에는 search_evidence(query, topic) 가 관련 chunk 를 반환한다.

이 스키마는 Strategy(질문 생성)와 Assessment(답변 평가)가 공통으로 소비한다.
→ 그래서 evidence 담당(A) 혼자 모양을 정하면 안 되고, B/D 와 합의해야 한다.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal
from typing import Any

from pydantic import BaseModel, Field, model_validator


# 근거 출처 종류. 새 소스(예: 블로그)를 붙일 일이 없으면 이 둘로 충분.
class SourceType(str, Enum):
    NOTION = "notion"
    GITHUB = "github"


class TopicCoverage(BaseModel):
    """한 기술 주제에 대해 evidence store 가 확보한 근거 요약.

    Evidence 파이프라인은 면접 전에 Notion/GitHub 자료를 청크로 만들고,
    각 청크의 topic/confidence 를 저장한다. Strategy 는 이 요약값을 보고
    어떤 주제를 우선 질문할지, 근거가 부족한 주제를 어떻게 다룰지 결정한다.
    """

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="해당 주제 청크들의 평균 신뢰도",
    )
    chunk_count: int = Field(
        ge=0,
        description="해당 주제에 연결된 근거 청크 개수",
    )


class EvidenceChunk(BaseModel):
    """면접 질문 생성과 답변 평가에 사용할 수 있는 최소 근거 단위.

    sources/extract/chunking 단계를 거친 결과물이며, store 는 이 모델을
    임베딩과 함께 저장한다. Strategy/Assessment 는 chunk_id 로 질문과
    평가가 어떤 원문 근거에 연결됐는지 추적한다.
    """
    chunk_id: str
    text: str  # 실제 근거 내용

    # --- 출처 메타데이터 ---
    source_type: SourceType
    source_url: str            # 원본 Notion 페이지 / GitHub 파일 URL
    topic: str                 # 기술 주제 (예: "JPA N+1", "JWT 인증")
    doc_type: str | None = None  # "주차정리"/"회고"/"코드"/"README" 등
    week: int | None = None      # 주차 (Notion 주차 기록일 때)
    date: str | None = None      # 날짜 (ISO 문자열, 예 "2026-03-01")

    # 신뢰도: 내용이 부족한 주제는 낮게 표시한다. (0.0 ~ 1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # GitHub 코드는 면접 근거의 출처와 사용자의 기여도를 함께 보존한다.
    # Notion 청크에는 기본값이 유지된다.
    file_path: str | None = None
    language: str | None = None
    ownership: Literal["user_touched", "repo_context"] | None = None
    commit_count: int = Field(default=0, ge=0)
    last_commit_sha: str | None = None


class RetrievalResult(BaseModel):
    """점수까지 필요한 retrieval 호출에서 사용할 검색 결과 모델.

    ``EvidenceChunk`` 자체에는 고정 메타데이터만 담고, 검색 쿼리마다 달라지는
    관련도 점수는 이 모델에서 분리해 표현한다. 현재 ``search_evidence`` 는
    chunk 리스트를 반환하지만, Assessment 가 근거 신뢰도와 검색 관련도를 함께
    쓰게 되면 이 모델을 반환 타입으로 전환할 수 있다.
    """
    chunk: EvidenceChunk
    score: float  # 쿼리와의 관련도 (재랭킹 점수). 높을수록 관련 큼.


class EvidenceSectionCandidate(BaseModel):
    """LLM이 문서 단위로 검토할 원문 섹션 후보.

    preview는 LLM 판단 비용을 줄이기 위한 입력이고, 실제 EvidenceChunk에는
    같은 section_id를 가진 원문 text 전체를 사용한다.
    """

    section_id: str
    heading: str | None = None
    preview: str
    topic_candidates: list[str] = Field(default_factory=list)


class EvidenceSectionDecision(BaseModel):
    """LLM이 개별 섹션에 부여한 검색용 주제와 신뢰도."""

    section_id: str
    topic: str
    confidence: float = Field(ge=0.0, le=1.0)
    doc_type: str | None = None


class EvidenceExtractionDecision(BaseModel):
    """문서 1건에 대한 LLM 구조화 추출 결과."""

    # legacy 필드는 기존 structured-output 테스트와 호환되도록 남긴다.
    topic: str | None = None
    doc_type: str | None = None
    valuable_section_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sections: list[EvidenceSectionDecision] = Field(default_factory=list)


class IndexFailure(BaseModel):
    """인덱싱 중 실패한 단일 source 또는 문서 처리 단계."""

    source_type: str
    source_url: str | None = None
    stage: str
    message: str


class IndexBuildResult(BaseModel):
    """면접 전 Evidence 인덱싱 작업의 실행 결과.

    CoverageMap은 Strategy가 소비하는 주제 요약이고, 이 모델은 API/status가
    사용자에게 보여줄 인덱싱 작업 결과를 함께 담는다.
    """

    status: Literal["success", "partial_failed", "failed"]
    coverage_map: "CoverageMap"
    raw_doc_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    failures: list[IndexFailure] = Field(default_factory=list)


class CoverageMap(BaseModel):
    """Strategy 가 소비하는 주제별 evidence 커버리지 맵.

    새 계약은 ``topic_coverage`` 에 주제별 평균 confidence 와 chunk 수를 함께
    담는다. 기존 Strategy 코드가 쓰는 ``topic_confidence`` 와 ``updated_at`` 은
    전환 기간 동안 호환되도록 유지한다.
    """

    topic_coverage: dict[str, TopicCoverage] = Field(default_factory=dict)
    updated_at: str | None = Field(None, description="커버리지 맵 생성/갱신 시점")

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_topic_confidence(cls, data: Any) -> Any:
        """기존 ``CoverageMap(topic_confidence={...})`` 생성자를 새 구조로 변환한다.

        Strategy/Assessment 쪽 코드가 한 번에 모두 바뀌기 전까지는 기존 생성
        방식을 허용해야 한다. 이 변환은 confidence 만 알 수 있는 legacy 입력에
        ``chunk_count=0`` 을 채워 새 ``topic_coverage`` 구조로 맞춘다.
        """
        if not isinstance(data, dict):
            return data
        if "topic_confidence" not in data or data.get("topic_coverage"):
            return data

        return {
            **data,
            "topic_coverage": {
                topic: {"confidence": confidence, "chunk_count": 0}
                for topic, confidence in data["topic_confidence"].items()
            },
        }

    @property
    def topic_confidence(self) -> dict[str, float]:
        """기존 Strategy 코드가 읽는 주제별 평균 confidence view."""
        return {
            topic: coverage.confidence
            for topic, coverage in self.topic_coverage.items()
        }

    def weak_topics(self, threshold: float = 0.4) -> list[str]:
        """근거 신뢰도가 threshold 미만인 주제를 반환한다.

        Strategy 는 이 목록을 보고 evidence 기반 프로젝트 질문을 피하거나,
        더 일반적인 기술 개념 질문으로 대체할 수 있다.
        """
        return [
            topic
            for topic, coverage in self.topic_coverage.items()
            if coverage.confidence < threshold
        ]

    def strong_topics(self, threshold: float = 0.7) -> list[str]:
        """근거 신뢰도가 threshold 이상인 주제를 반환한다.

        Strategy 는 이 목록을 사용자 자료에 기반한 우선 질문 후보로 사용할 수
        있다. chunk_count 는 이 메서드에서 직접 필터링하지 않고, 필요하면
        Strategy 쪽에서 추가 기준으로 함께 본다.
        """
        return [
            topic
            for topic, coverage in self.topic_coverage.items()
            if coverage.confidence >= threshold
        ]
