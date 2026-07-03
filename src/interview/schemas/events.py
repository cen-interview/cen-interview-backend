"""
Interviewer Agent 입력 이벤트 계약.

이 모듈은 채팅/음성 등 모든 입력 소스가 adapters.py를 거친 뒤
반드시 아래 5가지 타입 중 하나로 변환되어야 한다는 계약을 정의한다.

- Interviewer는 이 5가지 타입만 알면 되고, 입력이 어떤 모드(채팅/음성)에서
  왔는지는 알 필요가 없다.
- 모드별 반응 정책(예: 침묵 시 재전달할지 음성 안내를 할지)은 이 파일이 아니라
  SessionState 쪽에 정책 값으로 저장되고, Interviewer는 그 정책 값을 읽어 처리한다.
- 이 파일은 "모양"만 정의한다. session_id/question_id의 유효성 검증,
  빈 문자열 여부 등의 실제 검증은 validate_event 노드의 책임이다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from enum import Enum


class _EventBase(BaseModel):
    """모든 InterviewerEvent가 공유하는 공통 필드."""

    session_id: str
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnswerSubmitted(_EventBase):
    """채팅 제출 또는 음성 발화 종료(STT 완료) 시 발생.

    question_id는 반드시 현재 질문 ID와 일치해야 한다는 불변조건은
    이 파일이 아니라 이벤트 처리 로직(validate_event)에서 검증한다.
    """

    type: Literal["answer_submitted"] = "answer_submitted"
    question_id: str
    text: str


class ReplayRequested(_EventBase):
    """현재 질문을 다시 전달해달라는 요청.

    question_id를 생략하면 서버가 현재 질문 기준으로 처리한다.
    평가 결과나 asked_count에는 영향을 주지 않는다.
    """

    type: Literal["replay_requested"] = "replay_requested"
    question_id: str | None = None


class EndRequested(_EventBase):
    """사용자의 명시적 면접 종료 요청."""

    type: Literal["end_requested"] = "end_requested"


class SilenceDetected(_EventBase):
    """음성 입력에서 침묵이 감지되었을 때 발생.

    이 이벤트는 사실(침묵 지속 시간)만 전달한다.
    재전달할지 별도 음성 안내를 할지는 SessionState의 silence_policy를
    Interviewer가 읽어서 결정하며, 여기서 곧바로 오답으로 처리하지 않는다.
    """

    type: Literal["silence_detected"] = "silence_detected"
    silence_duration_seconds: float


class NoResponseTimeout(_EventBase):
    """일정 시간 동안 어떤 응답도 없을 때 발생.

    일시정지할지 종료할지는 SessionState의 timeout_policy를 따른다.
    """

    type: Literal["no_response_timeout"] = "no_response_timeout"
    elapsed_seconds: float | None = None


InterviewerEvent = Annotated[
    Union[
        AnswerSubmitted,
        ReplayRequested,
        EndRequested,
        SilenceDetected,
        NoResponseTimeout,
    ],
    Field(discriminator="type"),
]
"""Interviewer가 받아들이는 유일한 입력 타입.

# 현재 events 스키마의 의도와 다르게 events.py 에서 Mode 를 요청하는게 있음

class Mode(str, Enum):
    """
    [Stub] 면접 진행 모드 (채팅 / 음성)
    interviewer/session.py 에서 임포트하여 사용합니다.
    """
    CHAT = "chat"
    VOICE = "voice"
    
    # 팀원 C가 소문자로 썼을 경우를 대비해 소문자 속성도 방어적으로 추가해 둡니다.
    chat = "chat"
    voice = "voice"
알 수 없는 type 값이 들어오면 파싱 단계에서 ValidationError가 발생하며,
이는 도메인 오류 "알 수 없는 이벤트"로 변환되어야 한다 (10번 섹션).
"""