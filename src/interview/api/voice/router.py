"""음성 면접의 OpenAI Realtime 연결 정보를 제공하는 라우터."""

from fastapi import APIRouter, HTTPException, Response
from openai import OpenAIError

from interview.api.voice.schema import RealtimeTranscriptionTokenResponse
from interview.api.voice.service import create_realtime_transcription_client_secret


router = APIRouter(prefix="/interview/realtime-transcription", tags=["Voice"])


@router.post("/token", response_model=RealtimeTranscriptionTokenResponse)
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
