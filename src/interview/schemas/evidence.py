"""근거(Evidence) 관련 계약.

Evidence 인덱싱 파이프라인이 만들어 evidence_store 에 적재하고,
Strategy / Assessment 가 Retrieval Tool 로 꺼내 쓰는 데이터의 모양을 정의한다.
"""

from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    NOTION = "notion"
    GITHUB = "github"


class EvidenceChunk(BaseModel):
    """근거 한 조각. 검색 결과 1건이 이 모양으로 반환된다."""

    chunk_id: str
    text: str

    # 메타데이터 (설계 문서의 "출처 URL, 소스 유형, 주차, 날짜 ..." 에 해당)
    source_url: str
    source_type: SourceType
    topic: str                       # 기술 주제 (예: "JPA", "트랜잭션")
    week: int | None = None          # 주차별 학습 기록인 경우
    date: str | None = None          # ISO 날짜 문자열
    doc_type: str | None = None      # "개념정리" | "회고" | "코드" 등
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)  # 근거 신뢰도


class CoverageMap(BaseModel):
    """주제별 커버리지 맵.

    인덱싱이 끝나면 "어떤 주제가 충분히 다뤄졌고 어떤 주제가 빈약한지"를
    Strategy 가 참고한다. 내용이 부족한 주제는 낮은 신뢰도로 표시된다.
    """

    # topic -> 그 주제에 대한 평균/대표 신뢰도
    topic_confidence: dict[str, float] = Field(default_factory=dict)

    def weak_topics(self, threshold: float = 0.4) -> list[str]:
        """신뢰도가 낮은(=근거 부족) 주제 목록."""
        return [t for t, c in self.topic_confidence.items() if c < threshold]
