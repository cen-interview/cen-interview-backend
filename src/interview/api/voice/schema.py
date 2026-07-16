"""음성 API의 요청과 응답 모델."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

from interview.api.sessions.schema import VoiceDeliveryMetricsPayload
from interview.interviewer.turn_completion.buffer import VoiceTurnState


class TtsRequest(BaseModel):
    """면접관 발화를 음성으로 변환하기 위한 요청.

    Attributes:
        text:
            OpenAI Speech API가 음성으로 변환할 면접관 발화. API 입력
            한계와 동일하게 최대 4,096자로 제한한다.
    """

    text: str = Field(min_length=1, max_length=4096)

    @field_validator("text")
    @classmethod
    def strip_and_validate_text(cls, value: str) -> str:
        """음성으로 변환할 문장의 앞뒤 공백을 정리한다.

        Args:
            value:
                프론트에서 보낸 면접관 발화.

        Returns:
            앞뒤 공백을 제거한 면접관 발화.

        Raises:
            ValueError:
                공백을 제거한 뒤 발화 내용이 빈 문자열인 경우.
        """
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("음성으로 변환할 텍스트가 필요합니다.")
        return stripped_value


class RealtimeTranscriptionTokenResponse(BaseModel):
    """브라우저의 Realtime STT 연결에 사용할 단기 인증 정보.

    OpenAI 표준 API 키는 서버에만 보관하고, 브라우저에는 짧은 시간 동안
    유효한 client secret만 전달한다. 프론트는 이 값을 사용해 OpenAI
    Realtime API와 직접 WebRTC 연결을 만든다.

    Attributes:
        value:
            Realtime 연결 인증에 사용하는 단기 client secret.

        expires_at:
            client secret이 만료되는 시각을 나타내는 Unix timestamp.
    """

    value: str
    expires_at: int


class ConnectionAuthenticateMessage(BaseModel):
    """WebSocket 연결 직후 클라이언트가 보내는 인증 메시지.

    Access Token을 URL query string에 노출하지 않고 WebSocket 연결이 열린 뒤
    첫 JSON 메시지로 전달한다.

    Attributes:
        type:
            연결 인증 메시지를 나타내는 고정 discriminator.

        access_token:
            로그인 또는 refresh API에서 발급한 Access Token.
    """

    type: Literal["connection.authenticate"]
    access_token: str = Field(min_length=1, max_length=4096)


class _QuestionScopedClientMessage(BaseModel):
    """현재 질문 ID와 revision을 공통으로 전달하는 클라이언트 메시지."""

    question_id: str = Field(min_length=1, max_length=100)
    revision: int = Field(ge=0)

    @field_validator("question_id")
    @classmethod
    def normalize_question_id(cls, value: str) -> str:
        """질문 ID의 앞뒤 공백을 제거하고 빈 값을 거절한다.

        Args:
            value:
                클라이언트가 보낸 질문 ID.

        Returns:
            앞뒤 공백을 제거한 질문 ID.

        Raises:
            ValueError:
                공백 제거 후 질문 ID가 비어 있는 경우.
        """
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("question_id는 비어 있을 수 없습니다.")
        return normalized_value


class AnswerTranscriptUpdatedMessage(_QuestionScopedClientMessage):
    """클라이언트가 보내는 현재 음성 답변의 누적 전사문 최신본.

    Attributes:
        type:
            누적 전사문 갱신을 나타내는 고정 discriminator.

        text:
            delta가 아닌 현재 질문에서 지금까지 누적된 전사문 최신본.

        speech_active:
            메시지 생성 시점에 사용자가 실제로 발화 중인지 여부.

        segment_final:
            현재 STT 구간이 안정화된 최종 구간인지 여부.

        answer_duration_seconds:
            현재 답변 발화가 시작된 뒤 경과한 선택적 시간.

        metrics:
            말하기 속도와 필러 횟수 등 선택적인 음성 전달 지표.
    """

    type: Literal["answer.transcript.updated"]
    text: str = Field(max_length=50000)
    speech_active: bool
    segment_final: bool = False
    answer_duration_seconds: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    metrics: VoiceDeliveryMetricsPayload | None = None


class VoiceActivityChangedMessage(_QuestionScopedClientMessage):
    """전사문 변경 없이 현재 발화 상태만 전달하는 메시지.

    Attributes:
        type:
            발화 상태 변경을 나타내는 고정 discriminator.

        speech_active:
            사용자가 현재 발화 중이면 True.
    """

    type: Literal["voice.activity.changed"]
    speech_active: bool


class _ConfirmationScopedClientMessage(_QuestionScopedClientMessage):
    """활성 확인 질문 ID까지 공통으로 전달하는 클라이언트 메시지."""

    confirmation_id: str = Field(min_length=1, max_length=100)

    @field_validator("confirmation_id")
    @classmethod
    def normalize_confirmation_id(cls, value: str) -> str:
        """확인 질문 ID의 앞뒤 공백을 제거하고 빈 값을 거절한다.

        Args:
            value:
                클라이언트가 보낸 확인 질문 ID.

        Returns:
            앞뒤 공백을 제거한 확인 질문 ID.

        Raises:
            ValueError:
                공백 제거 후 confirmation ID가 비어 있는 경우.
        """
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("confirmation_id는 비어 있을 수 없습니다.")
        return normalized_value


class TurnConfirmationResponseReadyMessage(_ConfirmationScopedClientMessage):
    """프론트가 확인 응답 STT 수집 준비를 마쳤음을 전달한다.

    Attributes:
        type:
            확인 응답 준비 완료를 나타내는 고정 discriminator.

        playback_status:
            확인 질문 TTS가 정상 완료됐으면 completed, 텍스트 안내 후 STT만
            정상 준비됐으면 failed.
    """

    type: Literal["turn.confirmation.response.ready"]
    playback_status: Literal["completed", "failed"]


class TurnConfirmationResponseActivityChangedMessage(
    _ConfirmationScopedClientMessage
):
    """확인 응답 수집 중인 지원자의 발화 상태를 전달한다.

    Attributes:
        type:
            확인 응답 발화 상태 변경을 나타내는 고정 discriminator.

        speech_active:
            지원자가 확인 응답을 실제로 말하는 중이면 True.
    """

    type: Literal["turn.confirmation.response.activity.changed"]
    speech_active: bool


class TurnConfirmationRespondedMessage(_ConfirmationScopedClientMessage):
    """종료 확인 질문 이후의 지원자 응답 전사문을 전달한다.

    Attributes:
        type:
            확인 응답을 나타내는 고정 discriminator.

        confirmation_id:
            응답 대상인 활성 확인 질문의 고유 ID.

        response_revision:
            확인 응답 STT에 부여한 새 revision. 실질적인 추가 답변으로
            분류된 경우에만 buffer의 새 답변 revision으로 사용한다.

        text:
            확인 질문 이후 지원자가 말한 응답 원문.
    """

    type: Literal["turn.confirmation.responded"]
    response_revision: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_response_revision(self) -> "TurnConfirmationRespondedMessage":
        """확인 응답 revision이 원래 답변 revision보다 높은지 확인한다.

        Returns:
            revision 순서가 검증된 현재 확인 응답 메시지.

        Raises:
            ValueError:
                response_revision이 확인 기준 revision 이하인 경우.
        """
        if self.response_revision <= self.revision:
            raise ValueError("확인 응답 revision은 기준 revision보다 커야 합니다.")
        return self


VoiceTurnClientMessage = Annotated[
    AnswerTranscriptUpdatedMessage
    | VoiceActivityChangedMessage
    | TurnConfirmationResponseReadyMessage
    | TurnConfirmationResponseActivityChangedMessage
    | TurnConfirmationRespondedMessage,
    Field(discriminator="type"),
]
"""인증 이후 WebSocket에서 받을 수 있는 클라이언트 메시지 union."""

VOICE_TURN_CLIENT_MESSAGE_ADAPTER = TypeAdapter(VoiceTurnClientMessage)
"""클라이언트 JSON dict를 discriminator union으로 검증하는 adapter."""


class ConnectionReadyMessage(BaseModel):
    """인증과 현재 음성 질문 검증이 끝났음을 알리는 서버 메시지.

    Attributes:
        type:
            연결 준비 완료를 나타내는 고정 discriminator.

        session_id:
            현재 면접 세션 ID.

        question_id:
            서버가 현재 답변 대상으로 인식하는 질문 ID.

        revision:
            서버 buffer가 보유한 최신 전사문 revision.

        state:
            서버가 보유한 현재 음성 턴 상태.
    """

    type: Literal["connection.ready"] = "connection.ready"
    session_id: str
    question_id: str
    revision: int = Field(ge=0)
    state: VoiceTurnState


class TurnStateChangedMessage(BaseModel):
    """서버가 현재 음성 답변을 계속 수집함을 알리는 메시지.

    Attributes:
        type:
            음성 턴 상태 변경을 나타내는 고정 discriminator.

        question_id:
            상태가 변경된 질문 ID.

        revision:
            상태 판단에 사용된 최신 전사문 revision.

        state:
            현재 단계에서 전송하는 계속 듣기 상태.

        reason:
            상태를 유지하거나 되돌린 제한된 판단·제어 사유.
    """

    type: Literal["turn.state.changed"] = "turn.state.changed"
    question_id: str
    revision: int = Field(ge=0)
    state: Literal["listening"] = "listening"
    reason: str


class TurnConfirmationRequestedMessage(BaseModel):
    """답변 종료 여부를 묻는 고정 확인 질문 재생을 요청한다.

    Attributes:
        type:
            확인 질문 요청을 나타내는 고정 discriminator.

        confirmation_id:
            확인 질문 요청과 이후 취소·응답을 연결하는 고유 ID.

        question_id:
            종료 여부를 확인할 현재 질문 ID.

        revision:
            확인 판단에 사용한 전사문 revision.

        text:
            프론트가 TTS로 재생할 고정 확인 문구.

        ready_timeout_milliseconds:
            프론트가 확인 응답 수집 준비를 완료할 때까지의 제한 시간.

        response_timeout_milliseconds:
            준비 완료 후 실제 확인 응답을 기다리는 제한 시간.

        requires_ready_ack:
            응답 제한 시간을 시작하려면 ready 메시지가 필수인지 여부.
    """

    type: Literal["turn.confirmation.requested"] = "turn.confirmation.requested"
    confirmation_id: str
    question_id: str
    revision: int = Field(ge=0)
    text: str
    ready_timeout_milliseconds: int = Field(gt=0)
    response_timeout_milliseconds: int = Field(gt=0)
    requires_ready_ack: bool = True


class TurnConfirmationCancelledMessage(BaseModel):
    """진행 중이거나 준비 중인 확인 질문을 취소하도록 알린다.

    Attributes:
        type:
            확인 질문 취소를 나타내는 고정 discriminator.

        confirmation_id:
            취소할 확인 질문의 고유 ID.

        question_id:
            확인 질문이 속한 현재 질문 ID.

        reason:
            후보자가 다시 발화하는 등 확인 질문을 취소한 이유.
    """

    type: Literal["turn.confirmation.cancelled"] = "turn.confirmation.cancelled"
    confirmation_id: str
    question_id: str
    reason: str


class AnswerReactionMessage(BaseModel):
    """자동 제출 확정 직후 즉시 재생할 면접관 리액션을 전달한다.

    다음 질문 생성이 끝나기 전에 프론트가 리액션 TTS 재생을 시작할 수 있도록
    answer.committed보다 먼저 전송한다. 이 메시지를 받은 제출 건의
    answer.committed 세션 응답에서는 utterance_queue의 리액션 문장이 제거되고
    transcript의 마지막 면접관 발화와 last_utterance도 질문 본문으로 교체되므로
    음성과 화면 어느 쪽에서도 리액션이 두 번 나오지 않는다.

    Attributes:
        type:
            제출 확정 리액션을 나타내는 고정 discriminator.

        question_id:
            제출이 확정된, 방금 답변한 질문 ID.

        revision:
            제출에 사용한 최종 전사문 revision.

        text:
            프론트가 즉시 TTS로 재생할 중립 리액션 문구.
    """

    type: Literal["answer.reaction"] = "answer.reaction"
    question_id: str
    revision: int = Field(ge=0)
    text: str


class AnswerCommittedMessage(BaseModel):
    """현재 음성 답변 제출과 세션 진행 결과를 알리는 메시지.

    Attributes:
        type:
            답변 제출 완료를 나타내는 고정 discriminator.

        question_id:
            제출된 답변의 질문 ID.

        revision:
            최종 제출된 전사문 revision.

        completion_reason:
            자동 제출을 확정한 문맥 기반 사유.

        session:
            기존 세션 events API와 같은 형태의 최신 세션 응답.
    """

    type: Literal["answer.committed"] = "answer.committed"
    question_id: str
    revision: int = Field(ge=0)
    completion_reason: Literal[
        "semantic_complete",
        "explicit_finish",
        "user_confirmed",
        "listening_cutoff",
    ]
    session: dict[str, Any]


class VoiceTurnErrorMessage(BaseModel):
    """WebSocket 연결 또는 개별 이벤트 처리 오류를 전달한다.

    Attributes:
        type:
            오류 메시지를 나타내는 고정 discriminator.

        code:
            클라이언트가 복구 정책을 선택할 안정적인 오류 코드.

        recoverable:
            현재 연결을 유지하고 다음 이벤트를 처리할 수 있는지 여부.

        message:
            내부 예외 내용을 제외한 사용자 노출 가능 오류 설명.
    """

    type: Literal["error"] = "error"
    code: str
    recoverable: bool
    message: str
