"""면접 세션 API의 요청 모델."""

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator


class StartRequest(BaseModel):
    """면접 세션 생성 요청.

    Attributes:
        mode:
            면접 진행 모드. ``chat`` 또는 ``voice``.
    """

    mode: str


class SubmitEventPayload(BaseModel):
    """사용자가 명시적으로 제출한 답변 이벤트 payload.

    STT가 만든 전사문이나 채팅 입력을 면접 답변으로 확정할 때 사용한다.
    API는 앞뒤 공백을 제거한 뒤 비어 있지 않은 답변만 Interviewer 그래프에
    전달한다.

    Attributes:
        action:
            답변 제출을 나타내는 고정 action 값.

        text:
            사용자가 제출한 답변 또는 STT 최종 전사문.
    """

    action: Literal["submit"]
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """제출할 답변의 앞뒤 공백을 제거하고 빈 답변을 거절한다.

        Args:
            value:
                채팅 입력 또는 STT가 만든 원본 전사문.

        Returns:
            앞뒤 공백이 제거된 답변 문자열.

        Raises:
            ValueError:
                답변이 비어 있거나 공백으로만 구성된 경우.
        """
        normalized_text = value.strip()
        if not normalized_text:
            raise ValueError("답변 내용을 입력해 주세요.")
        return normalized_text


class SilenceEventPayload(BaseModel):
    """프론트가 감지한 연속 침묵 이벤트 payload.

    오디오 레벨 측정 자체는 프론트가 담당하고, 백엔드는 측정된 지속 시간이
    유효한 양수인지 확인한 뒤 Interviewer의 침묵 정책으로 전달한다.

    Attributes:
        action:
            침묵 감지를 나타내는 고정 action 값.

        silence_duration_seconds:
            음성이 감지되지 않은 연속 시간(초).
    """

    action: Literal["silence"]
    silence_duration_seconds: float = Field(gt=0, allow_inf_nan=False)


class EventRequest(BaseModel):
    """세션에 전달할 채팅 또는 음성 raw 이벤트 요청.

    Attributes:
        payload:
            입력 어댑터가 해석할 action과 관련 데이터를 담은 dict. session_id는
            URL 경로에서 받고 mode는 서버에 저장된 세션 상태를 사용한다.
    """

    payload: dict

    def to_adapter_payload(self) -> dict:
        """입력 어댑터에 전달할 이벤트 payload를 반환한다.

        submit 이벤트는 SubmitEventPayload로, silence 이벤트는
        SilenceEventPayload로 검증하고 정규화한다. 그 외 action은 다시 듣기·
        종료 처리를 담당하는 기존 어댑터가 해석할 수 있도록 원본 형태를
        유지한다. 전달 지표처럼 submit 모델에 아직 명시되지 않은 필드도 이후
        단계에서 사용할 수 있도록 보존한다.

        Returns:
            submit 텍스트 또는 침묵 지속 시간이 정규화된 이벤트 payload.
            검증 대상 action이 아니면 원본 payload의 복사본.

        Raises:
            ValueError:
                submit의 text 또는 silence의 지속 시간이 없거나 올바르지 않은
                경우.
        """
        normalized_payload = dict(self.payload)
        action = normalized_payload.get("action")

        if action == "submit":
            try:
                submitted = SubmitEventPayload.model_validate(normalized_payload)
            except ValidationError as exc:
                raise ValueError("제출할 답변 내용을 확인해 주세요.") from exc

            normalized_payload["action"] = submitted.action
            normalized_payload["text"] = submitted.text
            return normalized_payload

        if action == "silence":
            if "silence_duration_seconds" not in normalized_payload:
                normalized_payload["silence_duration_seconds"] = (
                    normalized_payload.get("silence_sec")
                )

            try:
                silence = SilenceEventPayload.model_validate(normalized_payload)
            except ValidationError as exc:
                raise ValueError("침묵 지속 시간을 확인해 주세요.") from exc

            normalized_payload["action"] = silence.action
            normalized_payload["silence_duration_seconds"] = (
                silence.silence_duration_seconds
            )
            normalized_payload.pop("silence_sec", None)
        return normalized_payload
