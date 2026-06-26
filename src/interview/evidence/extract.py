"""원본(RawDoc) → 근거 추출 + 메타데이터 부여.

기술 개념·코드 예시·프로젝트 경험·결정사항·구현 근거를 골라내고,
주제/주차/날짜/문서유형/신뢰도 메타데이터를 붙인다.
"""

from interview.evidence.sources import RawDoc
from interview.schemas.evidence import EvidenceChunk, SourceType


def extract_evidence(doc: RawDoc) -> list[EvidenceChunk]:
    """RawDoc 한 건에서 근거 후보들을 추출한다.

    여기서는 아직 임베딩하지 않는다. 텍스트 + 메타데이터까지만 만들고
    chunking 으로 넘긴다.

    TODO(담당 A):
      - 면접에 쓸 만한 부분만 선별 (목차/잡담 제외)
      - topic 분류 (LLM 또는 규칙 기반)
      - confidence 산정: 내용이 부족하면 낮게
    """
    raise NotImplementedError
