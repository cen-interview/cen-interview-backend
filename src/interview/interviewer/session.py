"""면접 세션 상태.

한 번의 면접 동안 유지되는 모든 상태. LangGraph 의 그래프 상태로도 쓰인다.
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from interview.schemas.question import Question
from interview.schemas.report import AnswerEvaluation, CompetencyModel
from interview.schemas.rubric import RubricCandidate, RubricSource


class Turn(BaseModel):
    """면접 대화에서 한 번의 발화를 나타내는 모델.

    Attributes:
        role:
            발화한 주체.
            - "interviewer": 면접관
            - "candidate": 지원자

        text:
            실제 발화 내용.

        question_id:
            해당 발화가 특정 질문과 연결될 경우 사용하는 질문 ID.

        kind:
            질문 유형.
            예: main, follow_up, challenge 등.

        created_at:
            발화가 생성된 시간.
            기본값은 현재 UTC 시간.
    """

    role: Literal["interviewer", "candidate"]
    text: str
    question_id: str | None = None
    kind: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SilencePolicy(BaseModel):
    """사용자의 침묵 상황을 처리하는 정책.

    Attributes:
        hint_threshold_seconds:
            사용자가 몇 초 이상 침묵했을 때 침묵 이벤트로 판단할지 정하는 기준값.

        first_action:
            첫 번째 침묵 이벤트 발생 시 수행할 동작.
            - "hint": 힌트 제공
            - "represent": 질문 재제시

        second_action:
            두 번째 침묵 이벤트 발생 시 수행할 동작.
            - "hint": 힌트 제공
            - "represent": 질문 재제시

        max_events_before_timeout:
            침묵 이벤트가 몇 번 누적되면 timeout으로 처리할지 정하는 기준값.
    """

    hint_threshold_seconds: float = 8.0
    first_action: Literal["hint", "represent"] = "hint"
    second_action: Literal["hint", "represent"] = "represent"
    max_events_before_timeout: int = 3


class TimeoutPolicy(BaseModel):
    """timeout 발생 시 처리 방식을 정하는 정책.

    Attributes:
        action:
            timeout 발생 시 수행할 동작.
            - "end": 면접 종료
            - "pause": 면접 일시정지
    """

    action: Literal["end", "pause"] = "end"

class SessionState(BaseModel):
    """면접 세션의 진행 상태를 저장하는 모델.

    현재 질문, 질문 개수, 기준 메인 질문, 대화 기록, 누적 평가 결과,
    침묵/타임아웃 정책, 마지막 처리 상태, 리포트 정보 등을 관리한다.

    Attributes:
        session_id:
            면접 세션을 구분하는 고유 ID.

        mode:
            면접 진행 모드.
            - "chat": 채팅 기반 면접
            - "voice": 음성 기반 면접

        max_questions:
            한 세션에서 진행할 최대 질문 수.

        current_question:
            현재 지원자에게 제시된 질문.
            아직 질문이 시작되지 않았다면 None일 수 있다.

        asked_count:
            현재까지 제시된 질문 수.

        main_question_id:
            현재 질문 세트의 기준이 되는 메인 질문 ID.
            후속 질문, 꼬리 질문, 챌린지 질문이 어떤 메인 질문에서 파생되었는지 추적할 때 사용한다.

        main_topic:
            현재 질문 세트의 기준 주제.
            예: Java, Spring, DB, 네트워크 등.

        transcript:
            면접관과 지원자의 전체 대화 기록.
            각 발화는 Turn 객체로 저장된다.

        pending_event:
            아직 처리되지 않은 입력 이벤트.
            그래프 흐름이나 다음 처리 단계에서 사용할 임시 이벤트를 저장할 때 사용한다.

        pending_delivery_metrics:
            현재 답변과 함께 전달된 부가 전달 지표.
            예: 말 빠르기, 침묵 시간, filler count 등.

        evaluations:
            지원자 답변에 대한 누적 평가 목록.

        competency:
            세션 전체 기준의 누적 역량 평가 모델.

        last_signal:
            마지막 답변 평가 결과 또는 라우팅 판단에 사용되는 신호.
            실제 타입은 프로젝트에서 정의한 평가 신호 타입을 사용한다.

        last_utterance:
            면접관이 마지막으로 생성한 발화 문장.

        utterance_queue:
            TTS가 순서대로 처리할 면접관 발화 목록. 질문이 있는 턴은 안내 또는
            리액션 문장과 질문 원문을 별도 항목으로 담고, 종료·일시정지처럼
            질문이 없는 턴은 안내 문장 하나만 담는다.

        turn_type:
            현재 턴의 성격.
            예: question, hint, represent, feedback 등.

        silence_policy:
            침묵 감지 시 어떤 방식으로 대응할지 정하는 정책.

        timeout_policy:
            무응답 타임아웃 발생 시 어떤 방식으로 처리할지 정하는 정책.

        timeout_action:
            마지막 타임아웃 이벤트를 처리한 결과. 그래프가 일시 정지 안내를
            제공할지 최종 리포트 생성으로 이동할지 결정할 때 사용한다.

        silence_count:
            현재 질문 또는 세션 흐름에서 누적된 침묵 이벤트 횟수.

        silence_action:
            마지막 침묵 이벤트를 처리한 결과. 그래프가 입력 대기, 힌트,
            질문 재제시 또는 타임아웃 중 다음 경로를 선택할 때 사용한다.

        challenge_used_in_set:
            현재 메인 질문 세트에서 challenge 질문을 이미 사용했는지 여부.

        derived_turn_count:
            현재 메인 질문 세트에서 생성한 파생 질문 수.

        max_derived_turns_per_set:
            하나의 메인 질문 세트에서 허용하는 최대 파생 질문 수.
            반복적인 꼬리 질문으로 면접이 끝나지 않는 상황을 방지한다.

        error:
            세션 처리 중 발생한 오류 메시지.
            오류가 없으면 None이다.

        report:
            면접 종료 후 생성된 최종 리포트 데이터.
            별도 FinalReport 모델이 있다면 dict 대신 해당 타입으로 교체할 수 있다.

        finished:
            면접 세션 종료 여부.
    """

    # 기본 세션 정보
    session_id: str
    mode: Literal["chat", "voice"] = "chat"
    max_questions: int = 10

    # 현재 질문 진행 상태
    current_question: Question | None = None
    asked_count: int = 0

    # 현재 질문 세트의 기준 메인 질문
    main_question_id: str | None = None
    main_topic: str | None = None
    challenge_used_in_set: bool = False
    derived_turn_count: int = 0
    max_derived_turns_per_set: int = 2

    # 대화 기록 및 입력 이벤트
    transcript: list[Turn] = Field(default_factory=list)
    pending_event: dict | None = None
    pending_delivery_metrics: dict | None = None

    # 누적 평가 및 마지막 평가 신호
    evaluations: list[AnswerEvaluation] = Field(default_factory=list)
    competency: CompetencyModel = Field(default_factory=CompetencyModel)
    last_signal: dict | None = None
    rubric_sources: list[RubricSource] = Field(default_factory=list)
    rubric_candidates: list[RubricCandidate] = Field(default_factory=list)
    rubric_share_status: Literal[
        "pending", "shared", "discarded", "not_available", "failed"
    ] | None = None
    rubric_share_approved: bool | None = None

    # 마지막 출력 및 턴 상태
    last_utterance: str = ""
    utterance_queue: list[str] = Field(default_factory=list)
    turn_type: str = "question"

    # 침묵/타임아웃 처리 정책
    silence_policy: SilencePolicy = Field(default_factory=SilencePolicy)
    timeout_policy: TimeoutPolicy = Field(default_factory=TimeoutPolicy)
    silence_count: int = 0
    silence_action: Literal["wait", "hint", "replay", "timeout"] | None = None
    timeout_action: Literal["pause", "end"] | None = None

    # 오류 및 결과 상태
    error: str | None = None
    report: dict | None = None
    finished: bool = False

    def is_done(self) -> bool:
        """면접 세션이 종료 조건에 도달했는지 확인한다."""

        return self.finished or self.asked_count >= self.max_questions
