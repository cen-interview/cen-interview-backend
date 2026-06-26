"""근거 청킹.

추출된 EvidenceChunk 텍스트가 너무 길면 임베딩에 적합한 크기로 쪼갠다.
메타데이터는 자식 청크에 그대로 상속한다.
"""

from interview.schemas.evidence import EvidenceChunk


def chunk(chunks: list[EvidenceChunk], max_chars: int = 1000) -> list[EvidenceChunk]:
    """긴 청크를 max_chars 기준으로 분할.

    TODO(담당 A):
      - 문단/코드 경계를 존중하는 분할 (문장 중간 자르지 않기)
      - 분할된 자식에 새 chunk_id 부여, 메타데이터 상속
    """
    raise NotImplementedError
