from pydantic import BaseModel, Field

from interview.schemas.events import InterviewerEvent


class DeliveryMetrics(BaseModel):
    """사용자의 답변 전달 방식과 관련된 측정값을 저장하는 모델.

    답변 내용 자체가 아니라 말하기 속도, 필러 표현 사용 횟수,
    답변 시간 등 음성 전달 품질과 관련된 정보를 관리한다.

    Attributes:
        speech_rate_wpm:
            분당 발화 단어 수(Words Per Minute).
            사용자가 얼마나 빠르거나 느리게 말했는지 나타낸다.

        filler_count:
            답변 중 사용된 필러 표현의 횟수.
            예: "음", "어", "그..." 등의 불필요한 추임새.

        duration_seconds:
            답변을 완료하는 데 걸린 전체 시간(초).
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


class AdaptedInput(BaseModel):
    """Interviewer Agent가 처리할 수 있도록 정규화된 입력 모델.

    외부에서 전달된 이벤트와 답변 전달 방식에 대한 부가 정보를
    하나의 입력 구조로 묶어 관리한다.

    Attributes:
        event:
            면접 진행 중 발생한 실제 이벤트.
            답변 제출, 침묵, 시간 초과 등의 상황을 나타낸다.

        delivery_metrics:
            사용자가 답변을 어떻게 전달했는지에 대한 부가 정보.
            음성 전달 정보가 없는 경우 None일 수 있다.
    """

    event: InterviewerEvent
    delivery_metrics: DeliveryMetrics | None = None


class ComposedUtterance(BaseModel):
    """LLM이 생성한 면접관 안내 문장을 저장하는 구조화 출력 모델.

    LLM은 Strategy가 만든 질문 본문을 생성하거나 수정하지 않고, 질문 앞에
    붙일 짧은 안내 문장만 반환한다. 실제 질문과의 결합은 Interviewer의
    compose_utterance 노드가 담당한다.

    Attributes:
        preamble:
            현재 상황에 맞는 짧은 한국어 존댓말 리액션. 질문 상황에서는
            한 문장으로 생성하며, 질문의 내용이나 핵심 표현은 포함하지 않는다.
    """

    preamble: str = Field(
        min_length=1,
        max_length=200,
        description="질문의 내용과 핵심 표현을 제외한 짧은 한국어 면접관 리액션.",
    )
