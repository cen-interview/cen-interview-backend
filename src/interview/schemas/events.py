"""Interviewer Agent 입력 이벤트 계약 모듈.

채팅과 음성 등 서로 다른 입력 소스에서 발생한 데이터를
Interviewer Agent가 처리할 수 있는 공통 이벤트 형태로 정의한다.

모든 입력은 adapter를 거친 뒤 다음 5가지 이벤트 중 하나로 변환된다.

- AnswerSubmitted
- ReplayRequested
- EndRequested
- SilenceDetected
- NoResponseTimeout

Interviewer Agent는 입력이 채팅 또는 음성 중 어떤 방식으로 들어왔는지
직접 알 필요 없이 정규화된 이벤트만 처리한다.

침묵이나 시간 초과에 대한 실제 대응 방식은 이 모듈에서 결정하지 않는다.
해당 정책은 SessionState에 저장되며, Interviewer Agent가 이를 읽어 처리한다.

이 모듈은 이벤트의 데이터 구조만 정의한다.
session_id, question_id의 유효성이나 빈 문자열 여부 등의 검증은
validate_event 노드에서 수행한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class _EventBase(BaseModel):
    """모든 Interviewer 이벤트가 공통으로 사용하는 기본 모델.

    Attributes:
        session_id:
            이벤트가 발생한 면접 세션의 ID.

        event_id:
            이벤트를 고유하게 식별하기 위한 ID.
            값을 전달하지 않으면 UUID가 자동으로 생성된다.

        occurred_at:
            이벤트가 발생한 시각.
            값을 전달하지 않으면 현재 UTC 시각이 자동으로 저장된다.
    """

    session_id: str
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class AnswerSubmitted(_EventBase):
    """사용자가 현재 질문에 대한 답변을 제출했을 때 발생하는 이벤트.

    채팅에서는 사용자가 답변을 제출했을 때 생성되며,
    음성에서는 발화가 종료되고 STT 변환이 완료된 뒤 생성된다.

    question_id가 현재 진행 중인 질문의 ID와 일치하는지는
    validate_event 노드에서 검증한다.

    Attributes:
        type:
            이벤트 종류를 식별하기 위한 고정 값.

        question_id:
            사용자가 답변한 질문의 ID.

        text:
            사용자가 제출한 답변 내용.
    """

    type: Literal["answer_submitted"] = "answer_submitted"
    question_id: str
    text: str


class ReplayRequested(_EventBase):
    """사용자가 현재 질문을 다시 전달해달라고 요청했을 때 발생하는 이벤트.

    질문을 다시 전달하는 동작은 답변 평가 결과나
    현재까지 제시된 질문 수에 영향을 주지 않는다.

    Attributes:
        type:
            이벤트 종류를 식별하기 위한 고정 값.

        question_id:
            다시 전달할 질문의 ID.
            값이 없으면 현재 진행 중인 질문을 기준으로 처리한다.
    """

    type: Literal["replay_requested"] = "replay_requested"
    question_id: str | None = None


class EndRequested(_EventBase):
    """사용자가 면접 종료를 명시적으로 요청했을 때 발생하는 이벤트.

    Attributes:
        type:
            이벤트 종류를 식별하기 위한 고정 값.
    """

    type: Literal["end_requested"] = "end_requested"


class SilenceDetected(_EventBase):
    """음성 입력 중 일정 시간 동안 침묵이 감지되었을 때 발생하는 이벤트.

    이 이벤트는 침묵이 발생했다는 사실과 지속 시간만 전달한다.
    질문을 다시 전달할지, 별도의 안내를 제공할지 등의 대응 방식은
    SessionState의 silence_policy를 기준으로 Interviewer Agent가 결정한다.

    침묵이 감지되었다는 이유만으로 사용자의 답변을 곧바로
    오답으로 평가하지 않는다.

    Attributes:
        type:
            이벤트 종류를 식별하기 위한 고정 값.

        silence_duration_seconds:
            감지된 침묵의 지속 시간(초).
    """

    type: Literal["silence_detected"] = "silence_detected"
    silence_duration_seconds: float


class NoResponseTimeout(_EventBase):
    """일정 시간 동안 사용자 응답이 없을 때 발생하는 이벤트.

    면접을 일시정지할지 종료할지 등의 대응 방식은
    SessionState의 timeout_policy를 기준으로 결정한다.

    Attributes:
        type:
            이벤트 종류를 식별하기 위한 고정 값.

        elapsed_seconds:
            사용자 응답 없이 경과한 시간(초).
            측정되지 않은 경우 None일 수 있다.
    """

    type: Literal["no_response_timeout"] = "no_response_timeout"
    elapsed_seconds: float | None = None


InterviewerEvent = Annotated[
    Union[
        AnswerSubmitted,
        ReplayRequested,
        EndRequested,
        SilenceDetected,
        NoResponseTimeout,
    ],
    Field(discriminator="type"),
]
"""Interviewer Agent가 처리할 수 있는 이벤트의 통합 타입.

type 필드를 discriminator로 사용하여 입력 데이터가
5가지 이벤트 모델 중 어떤 타입인지 자동으로 구분한다.
"""


class Mode(str, Enum):
    """면접 진행 방식을 나타내는 열거형.

    Attributes:
        CHAT:
            텍스트 채팅 방식의 면접 모드.

        VOICE:
            음성 기반 면접 모드.

        chat:
            소문자 속성 접근을 위한 CHAT의 별칭.

        voice:
            소문자 속성 접근을 위한 VOICE의 별칭.
    """

    CHAT = "chat"
    VOICE = "voice"

    chat = "chat"
    voice = "voice"
