"""음성 API의 요청과 응답 모델."""

from pydantic import BaseModel


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
