"""Interviewer adapter 단위 테스트.

이 파일은 채팅/음성 raw payload를 Interviewer 공통 입력인
AdaptedInput(event, delivery_metrics)으로 바꾸는 경계만 검증한다.

중요한 점:
    - Assessment, Strategy, evaluator는 전혀 사용하지 않는다.
    - 이벤트 데이터와 음성 전달 지표가 섞이지 않는지 확인한다.
    - 프론트 또는 음성 파이프라인이 보내는 action 값이 올바른 이벤트로
      정규화되는지만 본다.
"""

import pytest

from interview.interviewer.adapters import from_voice, from_chat
from interview.schemas.events import AnswerSubmitted


def test_voice_submit_separates_event_and_metrics():
    """음성 답변 제출 시 답변 이벤트와 음성 전달 지표를 분리한다."""
    adapted = from_voice(
        "session-1",
        "question-1",
        {
            "action": "submit",
            "text": "TCP는 연결 지향입니다",
            "speech_rate_wpm": 120.0,
        },
    )
    assert isinstance(adapted.event, AnswerSubmitted)
    assert adapted.event.text == "TCP는 연결 지향입니다"
    assert adapted.delivery_metrics.speech_rate_wpm == 120.0
    # 이벤트에는 지표가 없어야 한다 — 분리가 목적이므로
    assert not hasattr(adapted.event, "speech_rate_wpm")


def test_voice_silence_preserves_duration():
    """침묵 이벤트의 지속 시간이 Interviewer 이벤트에 그대로 전달된다."""
    adapted = from_voice(
        "session-1",
        "question-1",
        {"action": "silence", "silence_duration_seconds": 8.4},
    )
    assert adapted.event.silence_duration_seconds == 8.4


def test_unknown_action_raises():
    """지원하지 않는 action은 조용히 무시하지 않고 명시적으로 실패시킨다."""
    with pytest.raises(ValueError):
        from_voice("session-1", "question-1", {"action": "dance"})


def test_chat_submit_has_no_metrics():
    """채팅 답변에는 음성 전달 지표가 없으므로 delivery_metrics는 None이다."""
    adapted = from_chat(
        "session-1",
        "question-1",
        {"action": "submit", "text": "안녕하세요"},
    )
    assert adapted.delivery_metrics is None
