"""
Interviewer 입력 이벤트 (음성/채팅 통합)

adapters.py 가 모드별 raw 입력(제출버튼/발화종료감지/침묵 등)을
아래 이벤트 중 하나로 변환한다.
Interviewer 는 "모드"를 모른다. 오직 이 이벤트만 받아서 처리한다.
→ 그래서 음성/채팅 흐름 로직을 한 벌로 공유할 수 있다.

⚠️ 합의 포인트
  - 이벤트 종류를 추가/삭제할 때는 반드시 팀 합의.
  - 음성 전용 이벤트(침묵/타임아웃 등)는 채팅에선 절대 안 만들어진다.
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
    답변이 제출됨.
    채팅 = 제출 버튼 / 음성 = 발화 종료 감지(endpointing) → 둘 다 이걸로 변환된다.
    """
    type: Literal["answer_submitted"] = "answer_submitted"
    question_id: str
    text: str  # 사용자가 한 답변. 음성도 STT(음성→텍스트) 후 텍스트로 담는다.

    # --- (음성 전용) 전달력 보조 신호. 채팅 모드면 None. ---
    speech_rate_wpm: Optional[float] = None  # 말 속도 (분당 단어 수)
    filler_count: Optional[int] = None       # 군더더기("음", "어") 횟수


class EndRequested(BaseEvent):
    """종료 요청. 채팅 = 종료 버튼 / 음성 = '종료할게요'."""
    type: Literal["end_requested"] = "end_requested"


class SilenceDetected(BaseEvent):
    """(음성 전용) 일정 시간 침묵 = '막힘' 신호 → 힌트 질문으로 연결된다."""
    type: Literal["silence_detected"] = "silence_detected"
    silence_sec: float


class ReplayRequested(BaseEvent):
    """(음성 전용) '질문 다시 들려줘' → 직전 질문 TTS 재생."""
    type: Literal["replay_requested"] = "replay_requested"


class NoResponseTimeout(BaseEvent):
    """(음성 전용) 무응답 타임아웃 = 잠들기 대비. 우아하게 일시정지/종료."""
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