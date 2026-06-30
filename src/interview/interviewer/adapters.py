"""모드 어댑터: 음성/채팅의 raw 입력을 공통 InterviewEvent 로 변환한다.

설계의 핵심 분리점. 여기서 모드 차이를 흡수하면, Interviewer 흐름 로직은
모드를 몰라도 된다. 새 입력 채널이 생겨도 어댑터만 추가하면 된다.

  채팅 raw : 제출 버튼 payload / 종료 버튼      → AnswerSubmitted / EndRequested
  음성 raw : endpointing 결과 / 침묵 / STT 인텐트 / 타임아웃 → 각 이벤트
"""

from interview.schemas.events import (
    AnswerSubmitted,
    EndRequested,
    InterviewerEvent,
    Mode,
)


def from_chat(session_id: str, payload: dict) -> InterviewerEvent:
    """채팅 모드 raw payload → 이벤트.

    TODO(담당 C):
      - payload["action"] 가 "submit" 이면 AnswerSubmitted(answer_text=...)
      - "end" 이면 EndRequested
    """
    action = payload.get("action")
    if action == "submit":
        return AnswerSubmitted(
            session_id=session_id, mode=Mode.CHAT,
            answer_text=payload.get("text", ""),
        )
    if action == "end":
        return EndRequested(session_id=session_id, mode=Mode.CHAT)
    raise ValueError(f"unknown chat action: {action}")


def from_voice(session_id: str, payload: dict) -> InterviewerEvent:
    """음성 모드 raw payload → 이벤트.

    TODO(담당 C):
      - endpointing 으로 발화 종료 판정되면 AnswerSubmitted
        (음향 신호 + 의미적 완결성 함께 보고, 생각하느라 멈춤 vs 진짜 끝 구분)
      - 침묵 임계 초과 → SilenceDetected
      - STT 인텐트("다시 들려줘"/"다시 말할게요"/"종료") → Replay/Re-ask/End
      - 무응답 타임아웃 → NoResponseTimeout
      - delivery_metrics(말 속도/군더더기 등) 채워서 전달
    """
    raise NotImplementedError
