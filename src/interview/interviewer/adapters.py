"""모드 어댑터: 음성/채팅의 raw 입력을 공통 InterviewEvent로 변환한다.

설계의 핵심 분리점이다.
여기서 모드 차이를 흡수하면 Interviewer 흐름 로직은 입력 모드를 몰라도 된다.
새 입력 채널이 생겨도 어댑터 함수만 추가하면 된다.

변환 예시:
    채팅 raw:
        제출 버튼 payload / 종료 버튼
        -> AnswerSubmitted / EndRequested

    음성 raw:
        endpointing 결과 / 침묵 / STT 인텐트 / 타임아웃
        -> AnswerSubmitted / SilenceDetected / ReplayRequested / EndRequested / NoResponseTimeout
"""

from interview.schemas.events import (
    AnswerSubmitted,
    EndRequested,
    InterviewerEvent,
    NoResponseTimeout,
    ReplayRequested,
    SilenceDetected,
)


def from_chat(
    session_id: str,
    question_id: str,
    payload: dict,
) -> InterviewerEvent:
    """채팅 모드의 raw payload를 공통 InterviewEvent로 변환한다.

    Args:
        session_id:
            현재 면접 세션 ID.

        question_id:
            현재 답변 대상 질문 ID.

        payload:
            채팅 UI에서 전달된 원본 입력 데이터.
            action 값에 따라 이벤트 타입이 결정된다.

            지원하는 action:
                - "submit": 답변 제출
                - "end": 면접 종료 요청

    Returns:
        변환된 공통 InterviewEvent 객체.

    Raises:
        ValueError:
            지원하지 않는 채팅 action이 전달된 경우.
    """

    action = payload.get("action")

    if action == "submit":
        return AnswerSubmitted(
            session_id=session_id,
            question_id=question_id,
            text=payload.get("text", ""),
        )

    if action == "end":
        return EndRequested(session_id=session_id)

    raise ValueError(f"unknown chat action: {action}")


def from_voice(
    session_id: str,
    question_id: str,
    payload: dict,
) -> InterviewerEvent:
    """음성 모드의 raw payload를 공통 InterviewEvent로 변환한다.

    Args:
        session_id:
            현재 면접 세션 ID.

        question_id:
            현재 답변 대상 질문 ID.

        payload:
            음성 처리 파이프라인에서 전달된 원본 입력 데이터.
            STT 결과, 침묵 감지, 다시 듣기 요청, 종료 요청, 타임아웃 등을 포함한다.

            지원하는 action:
                - "submit": STT 답변 제출
                - "silence": 침묵 감지
                - "replay": 질문 다시 듣기 요청
                - "end": 면접 종료 요청
                - "timeout": 무응답 타임아웃

    Returns:
        변환된 공통 InterviewEvent 객체.

    Raises:
        ValueError:
            지원하지 않는 음성 action이 전달된 경우.
    """

    action = payload.get("action")

    if action == "submit":
        return AnswerSubmitted(
            session_id=session_id,
            question_id=question_id,
            text=payload.get("text", ""),
        )

    if action == "silence":
        return SilenceDetected(
            session_id=session_id,
            silence_duration_seconds=float(payload.get("silence_sec", 0.0)),
        )

    if action == "replay":
        return ReplayRequested(session_id=session_id)

    if action == "end":
        return EndRequested(session_id=session_id)

    if action == "timeout":
        return NoResponseTimeout(
            session_id=session_id,
            elapsed_seconds=float(payload.get("elapsed_sec", 0.0)),
        )

    raise ValueError(f"unknown voice action: {action}")