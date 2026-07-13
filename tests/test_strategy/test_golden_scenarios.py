"""파생 질문 골든 시나리오 테스트.

시나리오 문서의 4개 케이스에서 생성된 질문이 target을 실제로
겨냥하는지 확인한다. LLM 응답이 매번 달라지므로, target의 핵심
키워드가 질문 텍스트에 등장하는지로 느슨하게 검증한다.

실제 LLM을 호출하므로 평소 pytest 실행 대상에서 제외한다.
필요할 때만 아래처럼 마커를 지정해 수동으로 돌린다:

    uv run pytest tests/test_strategy/test_golden_scenarios.py -m golden -v
"""

import pytest

from interview.strategy.question_gen import (
    generate_challenge,
    generate_confirm_negative,
    generate_follow_up,
)

pytestmark = pytest.mark.golden


def test_golden_bonus_available_generates_follow_up_targeting_topic():
    """bonus_available → next_follow_up: target을 겨냥하는 꼬리 질문."""
    question = generate_follow_up(
        topic="JPA",
        parent_question_id="q-1",
        target="fetch join을 통한 N+1 해결 방법",
        answer_excerpt="N+1 문제는 지연 로딩 때문에 발생합니다",
    )
    assert "fetch join" in question.text or "N+1" in question.text


def test_golden_conflict_generates_confirm_negative_targeting_topic():
    """conflict → next_confirm_negative: 불일치 확인 질문."""
    question = generate_confirm_negative(
        topic="세션/토큰 인증",
        parent_question_id="q-1",
        target="세션 기반 인증과 토큰 기반 인증의 차이",
        answer_excerpt="세션은 상태를 유지하지 않는 방식입니다",
    )
    assert "세션" in question.text or "토큰" in question.text


def test_golden_misconception_generates_challenge_targeting_topic():
    """misconception → next_challenge: 오개념 지점을 파고드는 압박 질문."""
    question = generate_challenge(
        topic="JPA",
        parent_question_id="q-1",
        target="지연 로딩과 N+1의 관계",
        answer_excerpt="지연 로딩을 쓰면 N+1이 항상 줄어든다고 생각합니다",
        rationale=["지연 로딩과 N+1의 인과관계를 반대로 이해하고 있음"],
    )
    assert "지연 로딩" in question.text or "N+1" in question.text


def test_golden_troubleshooting_generates_follow_up_targeting_topic():
    """트러블슈팅 → next_follow_up: 트러블슈팅 관련 target을 겨냥."""
    question = generate_follow_up(
        topic="배포 트러블슈팅",
        parent_question_id="q-1",
        target="메모리 누수 원인 파악 과정",
        answer_excerpt="배포 후 메모리 사용량이 계속 늘어나는 문제가 있었습니다",
    )
    assert "메모리" in question.text