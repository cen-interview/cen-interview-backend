"""Interviewer 입력 이벤트.

설계의 핵심: 음성 모드와 채팅 모드의 서로 다른 입력 신호를 *같은 내부 이벤트*로
흡수한다. 모드 어댑터(api/adapters)가 raw 입력을 이 이벤트들로 변환하고,
Interviewer 의 흐름 로직은 모드와 무관하게 이 이벤트만 본다.

  채팅: 제출 버튼      → AnswerSubmitted
        종료 버튼      → EndRequested
  음성: 발화 종료 감지  → AnswerSubmitted
        침묵(막힘)     → SilenceDetected
        "다시 들려줘"   → ReplayRequested
        "종료할게요"    → EndRequested
        무응답 타임아웃  → NoResponseTimeout
"""

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, Field


class Mode(str, Enum):
    VOICE = "voice"
    CHAT = "chat"


class _BaseEvent(BaseModel):
    session_id: str
    mode: Mode


class AnswerSubmitted(_BaseEvent):
    """답변 완료. 채팅=제출 버튼, 음성=발화 종료 감지(endpointing)."""

    type: Literal["answer_submitted"] = "answer_submitted"
    answer_text: str
    # 음성 전달력 보조 신호 (채팅은 None). 말 속도/군더더기 등 원시 측정치.
    delivery_metrics: dict | None = None


class EndRequested(_BaseEvent):
    """면접 종료 요청. 채팅=종료 버튼, 음성='종료할게요' 인텐트."""

    type: Literal["end_requested"] = "end_requested"


class ReplayRequested(_BaseEvent):
    """음성: '질문 다시 들려줘'. 현재 질문을 다시 TTS 한다."""

    type: Literal["replay_requested"] = "replay_requested"


class SilenceDetected(_BaseEvent):
    """음성: 일정 시간 침묵 = 막힘. 힌트성 질문으로 연결."""

    type: Literal["silence_detected"] = "silence_detected"
    silence_seconds: float


class NoResponseTimeout(_BaseEvent):
    """음성: 무응답이 너무 길다(잠들기 대비). 우아하게 일시정지/종료."""

    type: Literal["no_response_timeout"] = "no_response_timeout"


# Interviewer 가 받는 모든 이벤트의 합. FastAPI/그래프에서 이걸로 받는다.
InterviewEvent = Union[
    AnswerSubmitted,
    EndRequested,
    ReplayRequested,
    SilenceDetected,
    NoResponseTimeout,
]
