"""next_question() 1회 호출 지연 측정 (10-3, 개발/검증용).

음성 모드 허용선(2~3초) 안에 들어오는지 확인한다.

실행:
    uv run python -m interview.strategy._measure_latency
"""

import time
from uuid import uuid4

from interview.evidence.store import get_store
from interview.schemas.evidence import EvidenceChunk, SourceType
from interview.strategy.agent import StrategyAgent

_TOPICS = ["FastAPI", "Docker", "JPA"]
_USER_ID = "latency-test-user"
_VOICE_MODE_THRESHOLD_SEC = 3.0


def _seed_evidence() -> None:
    store = get_store()
    chunks = [
        EvidenceChunk(
            chunk_id=str(uuid4()),
            text=f"{topic} 관련 학습 정리입니다.",
            source_type=SourceType.NOTION,
            source_url="https://example.com",
            topic=topic,
            confidence=0.8,
        )
        for topic in _TOPICS
        for _ in range(3)
    ]
    store.add_chunks(chunks, user_id=_USER_ID)


def measure(n: int = 5) -> None:
    _seed_evidence()
    coverage = get_store().build_coverage_map(user_id=_USER_ID)
    strategy = StrategyAgent(coverage=coverage, user_id=_USER_ID)

    durations = []
    for i in range(n):
        start = time.perf_counter()
        strategy.next_question(last_signal=None)
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        mark = "OK" if elapsed <= _VOICE_MODE_THRESHOLD_SEC else "SLOW"
        print(f"[{i + 1}] {elapsed:.2f}초 ({mark})")

    avg = sum(durations) / len(durations)
    print(f"\n평균: {avg:.2f}초")
    print(f"최대: {max(durations):.2f}초")
    print(f"최소: {min(durations):.2f}초")
    print(f"음성 모드 허용선({_VOICE_MODE_THRESHOLD_SEC}초) 초과 횟수: {sum(1 for d in durations if d > _VOICE_MODE_THRESHOLD_SEC)}/{n}")


if __name__ == "__main__":
    measure(5)