"""면접 세션 API의 요청 모델."""

from pydantic import BaseModel


class StartRequest(BaseModel):
    """면접 세션 생성 요청.

    Attributes:
        mode:
            면접 진행 모드. ``chat`` 또는 ``voice``.
    """

    mode: str


class EventRequest(BaseModel):
    """세션에 전달할 채팅 또는 음성 raw 이벤트 요청.

    Attributes:
        payload:
            입력 어댑터가 해석할 action과 관련 데이터를 담은 dict. session_id는
            URL 경로에서 받고 mode는 서버에 저장된 세션 상태를 사용한다.
    """

    payload: dict
