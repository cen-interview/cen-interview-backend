"""원본(RawDoc) → 근거 추출 + 메타데이터 부여.

기술 개념·코드 예시·프로젝트 경험·결정사항·구현 근거를 골라내고,
주제/주차/날짜/문서유형/신뢰도 메타데이터를 붙인다.
"""

from hashlib import sha1

from interview.evidence.sources import RawDoc
from interview.schemas.evidence import EvidenceChunk


def extract_evidence(doc: RawDoc) -> list[EvidenceChunk]:
    """RawDoc 한 건에서 근거 후보들을 추출한다.

    여기서는 아직 임베딩하지 않는다. 텍스트 + 메타데이터까지만 만들고
    chunking 으로 넘긴다. 현재 구현은 RawDoc 전체를 EvidenceChunk 1건으로
    넘기는 기본 추출기이며, 세부 선별/분류는 별도 추출 로직에서 확장한다.

    TODO(담당 A):
      - 면접에 쓸 만한 부분만 선별 (목차/잡담 제외)
      - topic 분류 (LLM 또는 규칙 기반)
      - confidence 산정: 내용이 부족하면 낮게
    """
    text = doc.raw_text.strip()
    if not text:
        return []

    topic = doc.meta.get("topic") or doc.title or "general"
    confidence = float(doc.meta.get("confidence", 0.7))
    confidence = min(max(confidence, 0.0), 1.0)

    return [
        EvidenceChunk(
            chunk_id=_chunk_id(doc),
            text=text,
            source_type=doc.source_type,
            source_url=doc.source_url,
            topic=topic,
            doc_type=doc.meta.get("doc_type"),
            week=doc.meta.get("week"),
            date=doc.meta.get("date"),
            confidence=confidence,
        )
    ]


def _chunk_id(doc: RawDoc) -> str:
    """원본 문서의 출처 정보를 바탕으로 안정적인 기본 chunk id 를 만든다."""
    digest = sha1(f"{doc.source_type}:{doc.source_url}:{doc.title}".encode()).hexdigest()
    return digest[:12]
