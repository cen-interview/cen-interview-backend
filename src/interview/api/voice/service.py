"""OpenAI Realtime 전사 연결 정보를 생성하는 서비스."""

from functools import lru_cache

from openai import OpenAI


@lru_cache
def get_openai_client() -> OpenAI:
    """환경 설정을 사용하는 OpenAI client를 프로세스에서 재사용한다.

    Returns:
        OPENAI_API_KEY 환경변수로 초기화된 OpenAI client.
    """
    return OpenAI()


def create_realtime_transcription_client_secret() -> tuple[str, int]:
    """한국어 Realtime 전사 세션용 단기 client secret을 발급한다.

    브라우저에는 표준 OpenAI API 키 대신 60초 동안 유효한 client secret만
    전달한다. gpt-realtime-whisper는 자동 turn detection을 사용하지 않으므로
    프론트가 음성 버퍼의 commit 시점을 직접 결정한다.

    Returns:
        client secret 값과 Unix timestamp 형식의 만료 시각.

    Raises:
        OpenAIError:
            OpenAI client 초기화 또는 client secret 발급에 실패한 경우.
    """
    token = get_openai_client().realtime.client_secrets.create(
        expires_after={
            "anchor": "created_at",
            "seconds": 60,
        },
        session={
            "type": "transcription",
            "audio": {
                "input": {
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                        "language": "ko",
                        "delay": "high",
                    },
                    "turn_detection": None,
                },
            },
        },
    )
    return token.value, token.expires_at
