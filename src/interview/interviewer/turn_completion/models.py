"""문맥 기반 음성 답변 완료 판단에 사용하는 데이터 모델."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TurnCompletionQuestionSnapshot(BaseModel):
    """완료 판단 시점의 현재 질문 정보를 나타낸다.

    Attributes:
        question_id:
            현재 지원자가 답변하고 있는 질문의 고유 ID.

        text:
            지원자에게 실제로 제시된 질문 본문.

        kind:
            main, follow_up, challenge 등 현재 질문의 역할.

        topic:
            현재 질문이 다루는 기술 또는 프로젝트 주제.
    """

    question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    topic: str = Field(min_length=1)


class TurnCompletionContextTurn(BaseModel):
    """완료 판단에 제한적으로 제공하는 최근 대화 한 턴을 나타낸다.

    Attributes:
        role:
            발화 주체. 면접관은 ``interviewer``, 지원자는 ``candidate``이다.

        text:
            해당 턴의 발화 내용.
    """

    role: Literal["interviewer", "candidate"]
    text: str = Field(min_length=1)


class TurnCompletionSnapshot(BaseModel):
    """LLM이 현재 음성 답변의 완료 여부를 판단할 입력 snapshot.

    부분 전사문은 확정된 면접 transcript가 아니므로 SessionState에 기록하지
    않고 이 모델에만 담는다. revision은 같은 질문 안에서 전사문이 갱신될
    때마다 증가하며, 늦게 도착한 판단 결과를 폐기하는 기준으로 사용한다.

    Attributes:
        session_id:
            현재 면접 세션의 고유 ID.

        question_id:
            현재 답변 대상 질문의 고유 ID.

        revision:
            현재 질문 안에서 누적 전사문 최신본의 단조 증가 버전.

        question:
            현재 질문의 본문, 종류와 주제를 담은 snapshot.

        current_answer:
            STT가 현재까지 만든 누적 답변 최신본. 확정 transcript가 아니다.

        recent_turns:
            판단에 필요한 최근 대화. 전체 transcript 대신 최대 두 턴만 담는다.

        speech_active:
            snapshot 생성 시점에 사용자가 실제로 발화 중인지 여부.

        segment_final:
            snapshot의 현재 STT 구간이 안정화된 최종 구간인지 여부. 이 값만으로
            의미적 답변 완료를 결정하지 않는다.

        answer_duration_seconds:
            현재 답변 발화가 시작된 뒤 경과한 선택적 시간. 음수가 아닌
            유한한 값이어야 한다.
    """

    session_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    question: TurnCompletionQuestionSnapshot
    current_answer: str = ""
    recent_turns: list[TurnCompletionContextTurn] = Field(
        default_factory=list,
        max_length=2,
    )
    speech_active: bool = False
    segment_final: bool = False
    answer_duration_seconds: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_question_id(self) -> "TurnCompletionSnapshot":
        """snapshot 최상위 질문 ID와 질문 정보의 ID가 같은지 확인한다.

        Returns:
            질문 ID 일관성이 검증된 현재 snapshot.

        Raises:
            ValueError:
                최상위 question_id와 question.question_id가 다른 경우.
        """
        if self.question_id != self.question.question_id:
            raise ValueError("완료 판단 snapshot의 question_id가 일치하지 않습니다.")
        return self


class TurnCompletionDecision(BaseModel):
    """현재 음성 답변의 문맥상 완료 여부를 나타낸다.

    답변의 정답 여부나 품질이 아니라 지원자가 현재 질문에 대해 하고 싶었던
    말을 마친 것으로 보이는지만 표현한다.

    Attributes:
        semantic_state:
            답변이 문맥상 완료, 미완료 또는 애매한 상태인지 나타낸다.

        linguistically_closed:
            문장이 접속 표현이나 미완성 절이 아니라 언어적으로 닫혔는지 여부.

        question_satisfied:
            답변이 정답인지를 뜻하지 않으며, 현재 질문에 대한 응답 의도가
            드러났는지 여부.

        continuation_expected:
            현재 문맥에서 지원자가 말을 이어갈 가능성.

        explicit_completion:
            ``이상입니다``처럼 명시적인 종료 표현이 포함됐는지 여부.

        recommended_action:
            자동 제출, 계속 듣기 또는 종료 확인 질문 중 권장 동작.

        confidence:
            판단 확신도. 0 이상 1 이하의 값이다.

        reason_code:
            판단의 주요 근거를 운영 정책에서 안정적으로 사용할 수 있도록
            제한된 코드로 표현한 값.
    """

    semantic_state: Literal["complete", "incomplete", "ambiguous"]
    linguistically_closed: bool
    question_satisfied: bool
    continuation_expected: Literal["low", "medium", "high"]
    explicit_completion: bool
    recommended_action: Literal[
        "auto_submit",
        "keep_listening",
        "ask_confirmation",
    ]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason_code: Literal[
        "explicit_finish",
        "complete_thought",
        "unfinished_clause",
        "enumeration_in_progress",
        "hesitation",
        "insufficient_context",
    ]

    @model_validator(mode="after")
    def validate_recommended_action(self) -> "TurnCompletionDecision":
        """의미 상태와 권장 동작이 명백히 모순되지 않는지 확인한다.

        Returns:
            의미 상태와 권장 동작의 일관성이 검증된 판단 결과.

        Raises:
            ValueError:
                미완료 답변을 자동 제출하거나 명확한 상태에서 확인 질문을
                권장하는 등 구조화 출력이 서로 모순되는 경우.
        """
        if self.recommended_action == "auto_submit" and self.semantic_state != "complete":
            raise ValueError("자동 제출은 complete 상태에서만 권장할 수 있습니다.")
        if (
            self.recommended_action == "ask_confirmation"
            and self.semantic_state != "ambiguous"
        ):
            raise ValueError("종료 확인은 ambiguous 상태에서만 권장할 수 있습니다.")
        if self.explicit_completion and self.semantic_state == "incomplete":
            raise ValueError("명시적 종료 표현과 incomplete 상태가 모순됩니다.")
        return self


class TurnCompletionResult(BaseModel):
    """입력 revision에 연결된 답변 완료 판단 결과.

    Attributes:
        question_id:
            판단 대상 질문의 고유 ID.

        revision:
            판단에 사용한 전사문 snapshot의 revision.

        decision:
            LLM 구조화 출력 또는 안전한 fallback으로 만든 완료 판단.
    """

    question_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    decision: TurnCompletionDecision


class ConfirmationIntentDecision(BaseModel):
    """답변 종료 확인에 대한 사용자의 응답 의도를 나타낸다.

    Attributes:
        intent:
            현재 답변을 제출하려는 ``finish``, 계속 생각하거나 말하려는
            ``continue``, 실질적인 추가 설명인 ``answer_content`` 또는
            해석할 수 없는 ``unknown`` 중 하나.

        answer_content:
            intent가 ``answer_content``일 때 기존 답변에 이어 붙일 원문 기반
            추가 내용. 그 외 intent에서는 None이어야 한다.

        confidence:
            확인 응답 의도 판단의 확신도. 0 이상 1 이하의 값이다.
    """

    intent: Literal["finish", "continue", "answer_content", "unknown"]
    answer_content: str | None = None
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_answer_content(self) -> "ConfirmationIntentDecision":
        """의도와 추가 답변 내용이 서로 일치하는지 확인한다.

        Returns:
            의도에 맞는 answer_content가 검증된 현재 판단 결과.

        Raises:
            ValueError:
                answer_content 의도에 실질적인 내용이 없거나, 다른 의도에
                answer_content가 포함된 경우.
        """
        if self.intent == "answer_content":
            if self.answer_content is None or not self.answer_content.strip():
                raise ValueError("추가 답변 의도에는 answer_content가 필요합니다.")
            self.answer_content = self.answer_content.strip()
            return self

        if self.answer_content is not None:
            raise ValueError(
                "finish, continue, unknown 의도에는 answer_content를 포함할 수 없습니다."
            )
        return self
