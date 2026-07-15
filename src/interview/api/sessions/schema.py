"""면접 세션 API의 요청 및 진행 상태 응답 모델."""

from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


_AUTO_SUBMIT_MIN_SILENCE_SECONDS = 2.0


class MainQuestionProgress(BaseModel):
    """현재 메인 질문 구간의 진행 정보를 나타낸다.

    Attributes:
        current:
            현재 진행 중인 메인 질문의 순번. 세션 시작 전에는 0이다.

        total:
            세션에서 물어볼 최대 메인 질문 수.
    """

    current: int = Field(ge=0)
    total: int = Field(ge=0)


class SessionProgress(BaseModel):
    """클라이언트에 전달할 면접 세션 진행 정보를 나타낸다.

    메인 질문 진행률과 실제 질문·답변 수를 분리한다. 실제 질문 수에는
    꼬리 질문과 힌트 등 파생 질문이 포함되지만, 같은 질문을 다시 들려주는
    replay는 중복으로 집계하지 않는다.

    Attributes:
        status:
            서버가 알고 있는 세션 상태. 진행 중이면 ``in_progress``, 종료되면
            ``completed``이다.

        main_question:
            현재 메인 질문 순번과 세션의 최대 메인 질문 수.

        asked_question_count:
            메인 질문과 파생 질문을 포함해 실제로 출제한 질문 수.

        answered_question_count:
            지원자의 답변이 제출된 질문 수.
    """

    status: Literal["in_progress", "completed"]
    main_question: MainQuestionProgress
    asked_question_count: int = Field(ge=0)
    answered_question_count: int = Field(ge=0)


class StartRequest(BaseModel):
    """면접 세션 생성 요청.

    Attributes:
        mode:
            면접 진행 모드. ``chat`` 또는 ``voice``.
    """

    mode: str


class VoiceDeliveryMetricsPayload(BaseModel):
    """음성 답변의 전달 방식을 나타내는 선택적 측정값.

    답변의 기술적 내용과 분리해 말하기 속도, 필러 표현 횟수, 실제 발화
    시간을 전달한다. metrics 영역을 보낸 경우 세 값 중 하나 이상은 존재해야
    한다.

    Attributes:
        speech_rate_wpm:
            분당 발화 단어 수. 음수가 아닌 유한한 값이어야 한다.

        filler_count:
            답변 중 감지된 필러 표현 횟수. 음수가 아니어야 한다.

        duration_seconds:
            답변의 실제 발화 시간. 자동 제출을 기다린 종료 침묵 시간은
            제외하며, 값이 있으면 0보다 커야 한다.
    """

    speech_rate_wpm: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    filler_count: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_metric_presence(self) -> "VoiceDeliveryMetricsPayload":
        """metrics 영역에 실제 측정값이 하나 이상 있는지 확인한다.

        Returns:
            하나 이상의 측정값을 가진 현재 metrics payload.

        Raises:
            ValueError:
                모든 측정값이 None인 경우.
        """
        if all(
            value is None
            for value in (
                self.speech_rate_wpm,
                self.filler_count,
                self.duration_seconds,
            )
        ):
            raise ValueError("metrics에는 하나 이상의 전달 지표가 필요합니다.")
        return self


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

        submission_type:
            사용자가 버튼으로 제출한 ``manual`` 또는 종료 침묵을 감지해
            제출한 ``auto``.

        silence_duration_seconds:
            자동 제출을 발생시킨 연속 침묵 시간. 수동 제출에서는 생략한다.

        metrics:
            음성 답변에서만 사용하는 선택적 전달 지표. 채팅 답변에서는
            생략할 수 있다.
    """

    action: Literal["submit"]
    text: str
    submission_type: Literal["manual", "auto"] = "manual"
    silence_duration_seconds: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    metrics: VoiceDeliveryMetricsPayload | None = None

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

    @model_validator(mode="after")
    def validate_auto_submission(self) -> "SubmitEventPayload":
        """자동 제출에 충분한 종료 침묵 시간이 포함됐는지 확인한다.

        Returns:
            자동 제출 조건이 검증된 현재 payload.

        Raises:
            ValueError:
                자동 제출인데 침묵 시간이 없거나 2초보다 짧은 경우.
        """
        if self.submission_type != "auto":
            return self

        if (
            self.silence_duration_seconds is None
            or self.silence_duration_seconds < _AUTO_SUBMIT_MIN_SILENCE_SECONDS
        ):
            raise ValueError("자동 제출에는 2초 이상의 종료 침묵이 필요합니다.")
        return self


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

        client_event_id:
            프론트가 한 번의 사용자 행동에 부여하는 멱등성 ID. 같은 ID가
            재전송되면 백엔드는 그래프를 다시 실행하지 않고 첫 결과를 반환한다.
    """

    payload: dict
    client_event_id: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("client_event_id")
    @classmethod
    def normalize_client_event_id(cls, value: str | None) -> str | None:
        """선택적인 클라이언트 이벤트 ID의 앞뒤 공백을 제거한다.

        Args:
            value:
                프론트가 생성한 이벤트 ID 또는 None.

        Returns:
            공백이 제거된 이벤트 ID 또는 None.

        Raises:
            ValueError:
                이벤트 ID가 공백으로만 구성된 경우.
        """
        if value is None:
            return None
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("client_event_id는 빈 문자열일 수 없습니다.")
        return normalized_value

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
            metric_keys = (
                "speech_rate_wpm",
                "filler_count",
                "duration_seconds",
            )
            if "metrics" not in normalized_payload and any(
                key in normalized_payload for key in metric_keys
            ):
                normalized_payload["metrics"] = {
                    key: normalized_payload.get(key)
                    for key in metric_keys
                    if key in normalized_payload
                }

            try:
                submitted = SubmitEventPayload.model_validate(normalized_payload)
            except ValidationError as exc:
                raise ValueError("제출할 답변 내용을 확인해 주세요.") from exc

            normalized_payload["action"] = submitted.action
            normalized_payload["text"] = submitted.text
            normalized_payload["submission_type"] = submitted.submission_type
            if submitted.silence_duration_seconds is not None:
                normalized_payload["silence_duration_seconds"] = (
                    submitted.silence_duration_seconds
                )
            if submitted.metrics is not None:
                normalized_payload["metrics"] = submitted.metrics.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            for key in metric_keys:
                normalized_payload.pop(key, None)
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
