"""모드 어댑터: 음성/채팅의 raw 입력을 Interviewer 공통 입력으로 변환한다.

이 모듈은 채팅과 음성처럼 서로 다른 입력 형식의 차이를 흡수한다.

각 입력 채널의 raw payload를 다음 두 가지 정보로 변환한다.

- event:
    면접 중 어떤 일이 발생했는지를 나타내는 InterviewerEvent.
    예: 답변 제출, 침묵 감지, 다시 듣기 요청, 종료 요청, 타임아웃.

- delivery_metrics:
    사용자가 답변을 어떻게 전달했는지에 대한 음성 전달 지표.
    음성 답변 제출 시에만 존재할 수 있으며,
    채팅 입력과 그 외 음성 이벤트에서는 None이다.

변환 예시:
    채팅 raw:
        제출 버튼 payload / 종료 버튼
        -> AdaptedInput(
            event=AnswerSubmitted / EndRequested,
            delivery_metrics=None,
        )

    음성 raw:
        STT 결과 / 침묵 / 다시 듣기 / 종료 / 타임아웃
        -> AdaptedInput(
            event=공통 InterviewerEvent,
            delivery_metrics=선택적 DeliveryMetrics,
        )

이렇게 모드별 차이를 어댑터에서 처리하면,
Interviewer Agent는 입력이 채팅에서 왔는지 음성에서 왔는지 알 필요 없이
AdaptedInput만 처리하면 된다.
"""

from interview.interviewer.models import AdaptedInput, DeliveryMetrics
from interview.schemas.events import (
    AnswerSubmitted,
    EndRequested,
    NoResponseTimeout,
    ReplayRequested,
    SilenceDetected,
)


def from_chat(
    session_id: str,
    question_id: str,
    payload: dict,
) -> AdaptedInput:
    """채팅 모드의 raw payload를 Interviewer 공통 입력으로 변환한다.

    채팅 입력에는 음성 전달 지표가 없으므로
    반환되는 AdaptedInput의 delivery_metrics는 항상 None이다.

    Args:
        session_id:
            현재 면접 세션 ID.

        question_id:
            현재 답변 대상 질문 ID.

        payload:
            채팅 UI에서 전달된 원본 입력 데이터.
            action 값에 따라 생성할 이벤트 타입이 결정된다.

            지원하는 action:
                - "submit": 답변 제출
                - "replay": 현재 질문 다시 보기 요청
                - "end": 면접 종료 요청

            예시:
                {
                    "action": "submit",
                    "text": "LCEL은 체인을 연결하는 문법입니다."
                }

    Returns:
        변환된 AdaptedInput 객체.

        event에는 변환된 InterviewerEvent가 저장되고,
        delivery_metrics는 항상 None이다.

    Raises:
        ValueError:
            지원하지 않는 채팅 action이 전달된 경우.
    """

    action = payload.get("action")

    if action == "submit":
        event = AnswerSubmitted(
            session_id=session_id,
            question_id=question_id,
            text=payload.get("text", ""),
        )

        return AdaptedInput(event=event)

    if action == "end":
        event = EndRequested(
            session_id=session_id,
        )

        return AdaptedInput(event=event)

    if action == "replay":
        event = ReplayRequested(
            session_id=session_id,
            question_id=question_id,
        )

        return AdaptedInput(event=event)

    raise ValueError(f"unknown chat action: {action}")


def from_voice(
    session_id: str,
    question_id: str,
    payload: dict,
) -> AdaptedInput:
    """음성 모드의 raw payload를 Interviewer 공통 입력으로 변환한다.

    답변 제출 이벤트에는 STT로 변환된 답변과 함께
    음성 전달 지표가 포함될 수 있다.

    음성 전달 지표 중 하나라도 payload에 존재하면
    DeliveryMetrics 객체를 생성한다.

    침묵, 다시 듣기, 종료, 타임아웃 이벤트에는
    음성 전달 지표가 필요하지 않으므로 delivery_metrics는 None이다.

    Args:
        session_id:
            현재 면접 세션 ID.

        question_id:
            현재 답변 대상 질문 ID.

        payload:
            음성 처리 파이프라인에서 전달된 원본 입력 데이터.

            지원하는 action:
                - "submit": STT 답변 제출
                - "silence": 침묵 감지
                - "replay": 질문 다시 듣기 요청
                - "end": 면접 종료 요청
                - "timeout": 무응답 타임아웃

            답변 제출 예시:
                {
                    "action": "submit",
                    "text": "LCEL은 체인을 연결하는 방식입니다.",
                    "metrics": {
                        "speech_rate_wpm": 120.0,
                        "filler_count": 2,
                        "duration_seconds": 15.4
                    }
                }

    Returns:
        변환된 AdaptedInput 객체.

        답변 제출 이벤트에서는 delivery_metrics가 포함될 수 있고,
        나머지 이벤트에서는 delivery_metrics가 None이다.

    Raises:
        ValueError:
            지원하지 않는 음성 action이 전달된 경우.
    """

    action = payload.get("action")

    if action == "submit":
        event = AnswerSubmitted(
            session_id=session_id,
            question_id=question_id,
            text=payload.get("text", ""),
        )

        metric_keys = (
            "speech_rate_wpm",
            "filler_count",
            "duration_seconds",
        )
        raw_metrics = payload.get("metrics")
        if raw_metrics is None and any(key in payload for key in metric_keys):
            raw_metrics = {
                key: payload.get(key)
                for key in metric_keys
                if key in payload
            }

        metrics = None
        if raw_metrics is not None:
            candidate_metrics = DeliveryMetrics.model_validate(raw_metrics)
            if candidate_metrics.model_dump(exclude_none=True):
                metrics = candidate_metrics

        return AdaptedInput(
            event=event,
            delivery_metrics=metrics,
        )

    if action == "silence":
        event = SilenceDetected(
            session_id=session_id,
            silence_duration_seconds=float(
                payload.get(
                    "silence_duration_seconds",
                    payload.get("silence_sec", 0.0),
                )
            ),
        )

        return AdaptedInput(event=event)

    if action == "replay":
        event = ReplayRequested(
            session_id=session_id,
        )

        return AdaptedInput(event=event)

    if action == "end":
        event = EndRequested(
            session_id=session_id,
        )

        return AdaptedInput(event=event)

    if action == "timeout":
        event = NoResponseTimeout(
            session_id=session_id,
            elapsed_seconds=float(
                payload.get(
                    "elapsed_seconds",
                    payload.get("elapsed_sec", 0.0),
                )
            ),
        )

        return AdaptedInput(event=event)

    raise ValueError(f"unknown voice action: {action}")
