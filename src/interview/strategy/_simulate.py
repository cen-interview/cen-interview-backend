"""Strategy 10문항 시뮬레이션 스크립트 (10-1, 개발/검증용).

정식 배포 코드가 아니라 눈으로 품질을 확인하기 위한 개발자용 도구.
필요 없어지면 삭제하거나 별도 scripts/ 폴더가 팀 차원에서 생기면 그때 이동.

실행:
    uv run python -m interview.strategy._simulate
"""

import random
from uuid import uuid4

from interview.evidence.store import get_store
from interview.schemas.evidence import EvidenceChunk, SourceType
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal
from interview.strategy.agent import StrategyAgent

_TOPICS = ["FastAPI", "Docker", "JPA", "Redis", "JWT 인증"]
_SIM_USER_ID = "sim-user"


def _seed_evidence() -> None:
    """시뮬레이션용 근거 청크를 store에 미리 채워둔다.

    실제 서비스에서는 사용자가 Notion/GitHub을 등록해야 면접이 시작되므로,
    근거가 0건인 극단적 상황은 정상 흐름에서 발생하지 않는다. 시뮬레이션도
    이를 반영해 각 주제마다 청크를 최소 2~3개씩 미리 넣어둔다.
    """
    store = get_store()
    chunks = []
    for topic in _TOPICS:
        for i in range(3):
            chunks.append(
                EvidenceChunk(
                    chunk_id=str(uuid4()),
                    text=f"{topic} 관련 학습 정리 {i + 1}: 실제 프로젝트에서 {topic}을 사용한 경험 예시입니다.",
                    source_type=SourceType.NOTION,
                    source_url="https://example.com/sim",
                    topic=topic,
                    confidence=random.uniform(0.6, 0.95),
                )
            )
    store.add_chunks(chunks, user_id=_SIM_USER_ID)


def _random_signal(question_id: str) -> AnswerQualitySignal:
    """무작위 quality를 가진 AnswerQualitySignal을 만든다 (시뮬레이션용)."""
    quality = random.choice(list(AnswerQuality))
    return AnswerQualitySignal(
        answer_id=f"a-{question_id}",
        question_id=question_id,
        quality=quality,
    )


def run_simulation(n: int = 10) -> None:
    """메인 질문 n개를 실제로 생성해보고 통계를 출력한다."""
    _seed_evidence()
    coverage = get_store().build_coverage_map(user_id=_SIM_USER_ID)

    print("=== 초기 CoverageMap ===")
    for t, c in coverage.topic_coverage.items():
        print(f"{t}: confidence={c.confidence:.2f}, chunk_count={c.chunk_count}")
    print()

    strategy = StrategyAgent(coverage=coverage, user_id=_SIM_USER_ID)

    last_signal = None
    questions = []

    for i in range(n):
        question = strategy.next_question(last_signal=last_signal)
        questions.append(question)
        print(f"[{i + 1}] ({question.difficulty.value}) {question.topic}: {question.text}")
        last_signal = _random_signal(question.question_id)

    print("\n=== 통계 ===")
    topics = [q.topic for q in questions]
    print(f"주제 중복도: {len(topics)}개 질문 중 고유 주제 {len(set(topics))}개")
    print(f"주제별 횟수: {strategy.state.topic_counts()}")
    print(f"난이도 분포: {strategy.state.difficulty_counts()}")

    print("\n=== 질문 텍스트 (유사도 눈검사용) ===")
    for q in questions:
        print(f"- {q.text}")


if __name__ == "__main__":
    run_simulation(10)