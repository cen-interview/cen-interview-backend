"""
Interviewer 입력 이벤트 (음성/채팅 통합)

adapters.py가 모드별 raw 입력을 아래 이벤트 중 하나로 변환한다.

예:
  - 채팅 제출 버튼
  - 음성 발화 종료 감지
  - 침묵 감지
  - 질문 다시 듣기 요청
  - 종료 요청

Interviewer는 입력 모드를 직접 알지 않고, 오직 InterviewerEvent만 받아 처리한다.
따라서 채팅/음성 면접 흐름을 하나의 로직으로 공유할 수 있다.

⚠️ 합의 포인트
  - 이벤트 종류를 추가/삭제할 때는 반드시 팀 합의가 필요하다.
  - 음성 전용 이벤트는 채팅 모드에서는 생성하지 않는다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from enum import Enum


class BaseEvent(BaseModel):
    """모든 이벤트 공통 필드."""
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AnswerSubmitted(BaseEvent):
    """
    답변 제출 이벤트.

    채팅:
      - 사용자가 제출 버튼을 누른 경우

    음성:
      - 발화 종료 감지 후 STT 결과가 생성된 경우
    """
    type: Literal["answer_submitted"] = "answer_submitted"
    question_id: str
    text: str  # 사용자가 한 답변. 음성도 STT(음성→텍스트) 후 텍스트로 담는다.

    # --- (음성 전용) 전달력 보조 신호. 채팅 모드면 None. ---
    speech_rate_wpm: Optional[float] = None  # 말 속도 (분당 단어 수)
    filler_count: Optional[int] = None       # 군더더기("음", "어") 횟수


class EndRequested(BaseEvent):
    """
    종료 요청 이벤트.

    채팅:
      - 종료 버튼 클릭

    음성:
      - 사용자가 "종료할게요"처럼 종료 의사를 말한 경우
    """
    type: Literal["end_requested"] = "end_requested"


class SilenceDetected(BaseEvent):
    """
    음성 전용 침묵 감지 이벤트.

    일정 시간 답변이 없을 때 생성된다.
    hint 질문을 별도로 두지 않는 현재 구조에서는
    Interviewer가 현재 질문을 다시 제시하거나,
    상황에 따라 꼬리/확인 질문으로 전환할 수 있다.
    """
    type: Literal["silence_detected"] = "silence_detected"
    silence_sec: float


class ReplayRequested(BaseEvent):
    """
    음성 전용 질문 재청취 이벤트.

    사용자가 "질문 다시 들려줘"라고 요청하면
    Interviewer가 현재 질문을 다시 반환하고,
    음성 어댑터가 이를 TTS로 재생한다.
    """
    type: Literal["replay_requested"] = "replay_requested"

class NoResponseTimeout(BaseEvent):
    """
    음성 전용 무응답 타임아웃 이벤트.

    장시간 응답이 없을 때 생성된다.
    세션을 우아하게 일시정지하거나 종료하는 데 사용한다.
    """
    type: Literal["no_response_timeout"] = "no_response_timeout"
    elapsed_sec: float


# 모든 이벤트의 합집합.
# Interviewer 는 handle(event: InterviewerEvent) 형태로 받아서
# event.type 값으로 분기한다.
InterviewerEvent = (
    AnswerSubmitted
    | EndRequested
    | SilenceDetected
    | ReplayRequested
    | NoResponseTimeout
)

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