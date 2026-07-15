"""음성 면접의 OpenAI Realtime 연결 정보와 TTS를 제공하는 라우터."""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from openai import OpenAIError

from interview.api.voice.schema import RealtimeTranscriptionTokenResponse, TtsRequest
from interview.api.voice.service import (
    create_realtime_transcription_client_secret,
    create_tts_audio_stream,
)
from interview.api.voice.websocket import router as voice_turn_websocket_router


router = APIRouter(prefix="/interview", tags=["Voice"])
router.include_router(voice_turn_websocket_router)


@router.post(
    "/tts",
    response_class=StreamingResponse,
    responses={200: {"content": {"audio/mpeg": {}}}},
)
def create_tts(request: TtsRequest) -> StreamingResponse:
    """면접관 발화를 OpenAI TTS 음성 스트림으로 반환한다.

    OpenAI 표준 API 키는 서버 환경변수에만 보관하고 프론트에는
    MP3 음성 청크만 전달한다. 생성된 면접 음성이 브라우저나
    중간 캐시에 남지 않도록 no-store 헤더를 설정한다.

    Args:
        request:
            음성으로 변환할 면접관 발화를 담은 요청.

    Returns:
        audio/mpeg 형식의 면접관 발화 스트림.

    Raises:
        HTTPException:
            OpenAI 설정 또는 upstream 요청 문제로 음성을 생성하지
            못한 경우 503 응답을 반환한다.
    """
    try:
        audio_stream = create_tts_audio_stream(request.text)
    except OpenAIError as exc:
        raise HTTPException(
            status_code=503,
            detail="면접관 음성을 생성할 수 없습니다.",
        ) from exc

    return StreamingResponse(
        audio_stream,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/realtime-transcription/token",
    response_model=RealtimeTranscriptionTokenResponse,
)
def create_realtime_transcription_token(
    response: Response,
) -> RealtimeTranscriptionTokenResponse:
    """OpenAI Realtime 전사용 단기 client secret을 발급한다.

    민감한 단기 인증 정보가 브라우저나 중간 캐시에 저장되지 않도록 응답에
    no-store 헤더를 추가한다. OpenAI client 초기화 또는 토큰 발급에 실패하면
    내부 예외 내용을 노출하지 않고 503 응답을 반환한다.

    Args:
        response:
            단기 인증 정보의 캐시를 막는 HTTP 헤더를 설정할 FastAPI 응답.

    Returns:
        Realtime WebRTC 연결에 사용할 client secret과 만료 시각.

    Raises:
        HTTPException:
            OpenAI 설정 또는 upstream 요청 문제로 client secret을 발급하지
            못한 경우 503 응답을 반환한다.
    """
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    try:
        value, expires_at = create_realtime_transcription_client_secret()
    except OpenAIError as exc:
        raise HTTPException(
            status_code=503,
            detail="Realtime 전사 연결 정보를 발급할 수 없습니다.",
        ) from exc

    return RealtimeTranscriptionTokenResponse(
        value=value,
        expires_at=expires_at,
    )
