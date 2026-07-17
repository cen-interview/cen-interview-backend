"""OpenAI Realtime 전사 연결 정보와 TTS 음성을 생성하는 서비스."""

from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from openai import OpenAI

from interview.config import settings


@lru_cache
def get_openai_client() -> OpenAI:
    """환경 설정을 사용하는 OpenAI client를 프로세스에서 재사용한다.

    Returns:
        OPENAI_API_KEY 환경변수로 초기화된 OpenAI client.
    """
    return OpenAI(api_key=settings.openai_api_key)


def create_tts_audio_stream(text: str) -> Iterator[bytes]:
    """면접관 발화를 MP3 음성 스트림으로 변환한다.

    OpenAI Speech API 연결을 응답 반환 전에 열어 upstream 설정
    오류를 HTTP 계층에서 처리할 수 있게 한다. 설정에 고정된 음성과
    한국인 남성 면접관 발화 지침을 모든 요청에 동일하게 적용한다.
    연결이 열리면 음성 데이터를 전체 메모리에 올리지 않고 청크 단위로
    반환한다.

    Args:
        text:
            음성으로 변환할 면접관 발화.

    Returns:
        MP3 음성 청크를 순차적으로 반환하는 iterator.

    Raises:
        OpenAIError:
            OpenAI client 초기화 또는 Speech API 요청에 실패한 경우.
    """
    response_context = get_openai_client().audio.speech.with_streaming_response.create(
        model=settings.openai_tts_model,
        voice=settings.openai_tts_voice,
        input=text,
        instructions=settings.openai_tts_instructions,
        response_format="mp3",
    )
    response = response_context.__enter__()
    return _iterate_tts_audio(response, response_context)


def _iterate_tts_audio(response: Any, response_context: Any) -> Iterator[bytes]:
    """OpenAI 응답을 읽고 스트림 연결을 안전하게 닫는다.

    Args:
        response:
            OpenAI SDK의 streaming binary response.

        response_context:
            response의 연결 생명주기를 관리하는 context manager.

    Yields:
        프론트에 즉시 전달할 MP3 음성 청크.
    """
    try:
        yield from response.iter_bytes()
    finally:
        response_context.__exit__(None, None, None)


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
