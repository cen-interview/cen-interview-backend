"""음성 API의 요청과 응답 모델."""

from pydantic import BaseModel, Field, field_validator


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
