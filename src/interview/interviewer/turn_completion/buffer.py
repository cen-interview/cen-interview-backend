"""제출 전 음성 답변의 실시간 상태와 상태 전이를 관리한다."""

from math import isfinite
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from interview.interviewer.models import DeliveryMetrics
from interview.interviewer.turn_completion.models import (
    ConfirmationIntentDecision,
    TurnCompletionDecision,
    TurnCompletionResult,
)


VoiceTurnState = Literal[
    "listening",
    "complete_candidate",
    "confirmation_pending",
    "confirming_end",
    "committing",
    "committed",
]
"""제출 전 음성 답변이 가질 수 있는 상태."""


class VoiceTurnBufferError(ValueError):
    """음성 턴 buffer 갱신 또는 상태 전이에 실패했을 때 발생하는 오류."""


class VoiceTurnQuestionMismatchError(VoiceTurnBufferError):
    """현재 질문과 다른 질문의 이벤트가 들어왔을 때 발생하는 오류."""


class VoiceTurnInvalidTransitionError(VoiceTurnBufferError):
    """현재 상태에서 허용되지 않는 상태 전이를 요청했을 때 발생하는 오류."""


class VoiceTurnAlreadyCommittedError(VoiceTurnBufferError):
    """이미 제출된 음성 턴을 다시 변경하려 할 때 발생하는 오류."""


class VoiceTurnBuffer(BaseModel):
    """아직 제출되지 않은 현재 음성 답변의 실시간 상태를 보관한다.

    부분 전사문과 완료 판단은 확정된 면접 도메인 상태가 아니므로
    SessionState와 분리한다. 이 모델에는 JSON 직렬화가 가능한 데이터만
    보관하고, task, lock, WebSocket 연결 객체는 registry와 coordinator가
    별도로 관리한다.

    Attributes:
        session_id:
            현재 면접 세션의 고유 ID.

        question_id:
            현재 답변 대상 질문의 고유 ID.

        revision:
            같은 질문 안에서 누적 전사문 최신본의 단조 증가 버전.

        answer_text:
            STT가 현재까지 만든 누적 답변 최신본. 제출 전까지 확정 transcript가
            아니다.

        speech_active:
            사용자가 현재 실제로 발화 중인지 여부.

        segment_final:
            현재 STT 구간이 안정화된 최종 구간인지 여부.

        answer_duration_seconds:
            현재 답변 발화가 시작된 뒤 경과한 선택적 시간.

        latest_delivery_metrics:
            현재까지 전달된 말하기 속도, 필러 횟수와 실제 발화 시간의 최신
            선택적 관찰 값.

        state:
            현재 음성 답변의 수집, 확인 또는 제출 진행 상태.

        latest_decision_revision:
            latest_decision을 생성할 때 사용한 전사문 revision.

        latest_decision:
            현재까지 반영된 가장 최근의 문맥상 완료 판단.

        confirmation_count:
            현재 질문에서 실제로 시작한 종료 확인 질문 횟수.

        committed_revision:
            기존 답변 제출 경로로 최종 제출을 완료한 revision.
    """

    session_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    revision: int = Field(default=0, ge=0)
    answer_text: str = ""
    speech_active: bool = False
    segment_final: bool = False
    answer_duration_seconds: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    latest_delivery_metrics: DeliveryMetrics | None = None
    state: VoiceTurnState = "listening"
    latest_decision_revision: int | None = Field(default=None, ge=0)
    latest_decision: TurnCompletionDecision | None = None
    confirmation_count: int = Field(default=0, ge=0)
    committed_revision: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_state_consistency(self) -> "VoiceTurnBuffer":
        """직렬화된 buffer 필드 조합이 서로 일치하는지 확인한다.

        Returns:
            revision, 판단 결과와 commit 상태의 일관성이 검증된 현재 buffer.

        Raises:
            ValueError:
                판단 revision이나 commit revision이 현재 revision보다 크거나,
                상태와 관련 필드가 서로 모순되는 경우.
        """
        if (self.latest_decision is None) != (self.latest_decision_revision is None):
            raise ValueError("완료 판단과 판단 revision은 함께 존재해야 합니다.")
        if (
            self.latest_decision_revision is not None
            and self.latest_decision_revision > self.revision
        ):
            raise ValueError("완료 판단 revision은 현재 revision보다 클 수 없습니다.")
        if self.committed_revision is not None and self.committed_revision > self.revision:
            raise ValueError("제출 revision은 현재 revision보다 클 수 없습니다.")
        if self.state == "committed":
            if self.committed_revision != self.revision:
                raise ValueError("committed 상태는 현재 revision의 제출 기록이 필요합니다.")
            if self.speech_active:
                raise ValueError("committed 상태에서는 발화 중일 수 없습니다.")
        elif self.committed_revision is not None:
            raise ValueError("committed 상태가 아니면 committed_revision을 가질 수 없습니다.")
        return self

    def update_transcript(
        self,
        *,
        question_id: str,
        revision: int,
        text: str,
        speech_active: bool,
        segment_final: bool,
        answer_duration_seconds: float | None = None,
        delivery_metrics: DeliveryMetrics | None = None,
    ) -> bool:
        """더 높은 revision의 누적 전사문 snapshot을 반영한다.

        현재 revision 이하의 이벤트는 재전송되거나 늦게 도착한 것으로 보고
        상태를 변경하지 않는다. 새 전사문은 완료 후보, 확인 준비와 진행 중인
        commit을 취소하고 다시 listening 상태로 되돌린다.

        Args:
            question_id:
                전사문이 속한 질문의 고유 ID.

            revision:
                질문 안에서 단조 증가하는 전사문 버전.

            text:
                delta가 아닌 현재까지 누적된 답변 최신본.

            speech_active:
                snapshot 시점에 사용자가 발화 중인지 여부.

            segment_final:
                현재 STT 구간이 안정화됐는지 여부.

            answer_duration_seconds:
                답변 시작 후 경과한 선택적 시간. 없으면 기존 관찰 값을
                유지한다.

            delivery_metrics:
                현재 답변의 선택적 음성 전달 지표. 없으면 기존 최신 지표를
                유지한다.

        Returns:
            새 revision을 반영했으면 True. 오래된 revision을 무시했으면 False.

        Raises:
            VoiceTurnQuestionMismatchError:
                현재 질문과 다른 질문 ID가 전달된 경우.

            VoiceTurnAlreadyCommittedError:
                이미 제출이 완료된 buffer를 변경하려는 경우.

            VoiceTurnBufferError:
                답변 경과 시간이 음수이거나 유한하지 않은 경우.
        """
        self._validate_question_id(question_id)
        self._ensure_not_committed()
        if revision <= self.revision:
            return False
        if answer_duration_seconds is not None and (
            answer_duration_seconds < 0 or not isfinite(answer_duration_seconds)
        ):
            raise VoiceTurnBufferError(
                "답변 경과 시간은 0 이상의 유한한 값이어야 합니다."
            )

        self.revision = revision
        self.answer_text = text
        self.speech_active = speech_active
        self.segment_final = segment_final
        if answer_duration_seconds is not None:
            self.answer_duration_seconds = answer_duration_seconds
        if delivery_metrics is not None:
            self.latest_delivery_metrics = delivery_metrics
        self.latest_decision_revision = None
        self.latest_decision = None
        self.state = "listening"
        return True

    def update_speech_activity(
        self,
        *,
        question_id: str,
        revision: int,
        speech_active: bool,
    ) -> bool:
        """전사문을 바꾸지 않고 현재 발화 상태만 갱신한다.

        activity 이벤트는 현재 전사 revision과 같을 때만 반영한다. 사용자가
        다시 말하기 시작하면 완료 후보와 확인 상태를 취소하고 listening으로
        복귀한다.

        Args:
            question_id:
                activity 이벤트가 속한 질문의 고유 ID.

            revision:
                activity 이벤트가 참조하는 현재 전사문 revision.

            speech_active:
                사용자가 현재 발화 중인지 여부.

        Returns:
            현재 revision의 activity를 반영했으면 True. revision이 다르면 False.

        Raises:
            VoiceTurnQuestionMismatchError:
                현재 질문과 다른 질문 ID가 전달된 경우.

            VoiceTurnAlreadyCommittedError:
                이미 제출이 완료된 buffer를 변경하려는 경우.
        """
        self._validate_question_id(question_id)
        self._ensure_not_committed()
        if revision != self.revision:
            return False

        self.speech_active = speech_active
        if speech_active:
            self.segment_final = False
            self.latest_decision_revision = None
            self.latest_decision = None
            self.state = "listening"
        return True

    def record_decision(self, result: TurnCompletionResult) -> bool:
        """현재 질문과 revision에 해당하는 최신 완료 판단을 기록한다.

        Args:
            result:
                판단 대상 질문 ID와 revision이 연결된 완료 판단 결과.

        Returns:
            현재 snapshot의 판단을 기록했으면 True. 오래된 revision이거나
            제출 중인 상태여서 무시했으면 False.

        Raises:
            VoiceTurnQuestionMismatchError:
                판단 결과의 질문 ID가 현재 질문과 다른 경우.
        """
        self._validate_question_id(result.question_id)
        if self.state != "listening":
            return False
        if result.revision != self.revision:
            return False

        self.latest_decision_revision = result.revision
        self.latest_decision = result.decision
        return True

    def mark_complete_candidate(self, *, expected_revision: int) -> None:
        """현재 완료 판단을 자동 제출 전 완료 후보 상태로 전환한다.

        Args:
            expected_revision:
                완료 후보로 지정할 최신 전사문 revision.

        Raises:
            VoiceTurnInvalidTransitionError:
                listening 상태가 아니거나, 사용자가 발화 중이거나, 최신 판단이
                해당 revision의 auto_submit 결정이 아닌 경우.
        """
        self._require_state("listening")
        self._validate_actionable_decision(
            expected_revision=expected_revision,
            recommended_action="auto_submit",
        )
        if self.speech_active:
            raise VoiceTurnInvalidTransitionError(
                "발화 중에는 완료 후보 상태로 전환할 수 없습니다."
            )
        if not self.answer_text.strip():
            raise VoiceTurnInvalidTransitionError("빈 답변은 완료 후보가 될 수 없습니다.")
        self.state = "complete_candidate"

    def mark_confirmation_pending(
        self,
        *,
        expected_revision: int,
        max_confirmations: int,
    ) -> bool:
        """현재 애매한 판단을 종료 확인 준비 상태로 전환한다.

        Args:
            expected_revision:
                확인 질문을 준비할 최신 전사문 revision.

            max_confirmations:
                현재 질문에서 허용할 최대 종료 확인 횟수.

        Returns:
            확인 준비 상태로 전환했으면 True. 이미 횟수 제한에 도달했으면
            listening을 유지하고 False.

        Raises:
            VoiceTurnInvalidTransitionError:
                최대 횟수가 0 이하이거나 현재 판단과 상태가 확인 조건에 맞지
                않는 경우.
        """
        if max_confirmations <= 0:
            raise VoiceTurnInvalidTransitionError(
                "최대 확인 질문 횟수는 0보다 커야 합니다."
            )
        self._require_state("listening")
        self._validate_actionable_decision(
            expected_revision=expected_revision,
            recommended_action="ask_confirmation",
        )
        if self.speech_active:
            raise VoiceTurnInvalidTransitionError(
                "발화 중에는 종료 확인을 준비할 수 없습니다."
            )
        if self.confirmation_count >= max_confirmations:
            return False

        self.state = "confirmation_pending"
        return True

    def begin_confirmation(self, *, max_confirmations: int) -> None:
        """준비된 종료 확인 질문을 시작하고 사용 횟수를 증가시킨다.

        Args:
            max_confirmations:
                현재 질문에서 허용할 최대 종료 확인 횟수.

        Raises:
            VoiceTurnInvalidTransitionError:
                확인 준비 상태가 아니거나 최대 횟수에 도달한 경우.
        """
        self._require_state("confirmation_pending")
        if max_confirmations <= 0 or self.confirmation_count >= max_confirmations:
            raise VoiceTurnInvalidTransitionError("종료 확인 질문 횟수 제한에 도달했습니다.")

        self.confirmation_count += 1
        self.state = "confirming_end"

    def apply_confirmation_intent(
        self,
        *,
        question_id: str,
        decision: ConfirmationIntentDecision,
        new_revision: int | None = None,
    ) -> None:
        """확인 응답 의도를 현재 답변과 상태에 반영한다.

        finish 응답은 확인 문구를 답변에 넣지 않고 commit을 시작한다. continue와
        unknown은 기존 답변을 유지한 채 listening으로 돌아간다. answer_content는
        실질적인 추가 내용만 연결하고 더 높은 revision으로 갱신한다.

        Args:
            question_id:
                확인 응답이 속한 현재 질문 ID.

            decision:
                확인 응답 의도와 선택적인 추가 답변 내용.

            new_revision:
                answer_content를 연결할 때 사용할 새 revision. 다른 의도에서는
                사용하지 않는다.

        Raises:
            VoiceTurnQuestionMismatchError:
                현재 질문과 다른 질문 ID가 전달된 경우.

            VoiceTurnInvalidTransitionError:
                confirming_end 상태가 아니거나 추가 내용에 유효한 새 revision이
                없는 경우.
        """
        self._validate_question_id(question_id)
        self._require_state("confirming_end")

        if decision.intent == "finish":
            self.begin_commit(
                question_id=question_id,
                expected_revision=self.revision,
            )
            return

        if decision.intent in {"continue", "unknown"}:
            self.resume_listening()
            return

        if new_revision is None or new_revision <= self.revision:
            raise VoiceTurnInvalidTransitionError(
                "추가 답변에는 현재보다 높은 새 revision이 필요합니다."
            )

        additional_content = decision.answer_content or ""
        if not additional_content.strip():
            raise VoiceTurnInvalidTransitionError("연결할 추가 답변 내용이 없습니다.")
        self.answer_text = " ".join(
            part for part in (self.answer_text.strip(), additional_content.strip()) if part
        )
        self.revision = new_revision
        self.speech_active = False
        self.segment_final = False
        self.latest_decision_revision = None
        self.latest_decision = None
        self.state = "listening"

    def resume_listening(self) -> None:
        """현재 제출 전 상태를 취소하고 답변 수집 상태로 돌아간다.

        Raises:
            VoiceTurnAlreadyCommittedError:
                이미 제출이 완료된 buffer를 다시 수집 상태로 바꾸려는 경우.
        """
        self._ensure_not_committed()
        self.latest_decision_revision = None
        self.latest_decision = None
        self.state = "listening"

    def begin_commit(self, *, question_id: str, expected_revision: int) -> str:
        """현재 질문과 revision을 검증하고 최종 제출 상태로 진입한다.

        Args:
            question_id:
                제출할 답변의 질문 ID.

            expected_revision:
                제출할 답변 snapshot의 revision.

        Returns:
            기존 제출 경로에 전달할 앞뒤 공백이 제거된 답변 문자열.

        Raises:
            VoiceTurnQuestionMismatchError:
                현재 질문과 다른 질문 ID가 전달된 경우.

            VoiceTurnInvalidTransitionError:
                완료 후보나 종료 확인 상태가 아니거나 revision과 답변 내용이
                제출 조건에 맞지 않는 경우.

            VoiceTurnAlreadyCommittedError:
                이미 제출이 완료된 경우.
        """
        self._validate_question_id(question_id)
        self._ensure_not_committed()
        if self.state not in {"complete_candidate", "confirming_end"}:
            raise VoiceTurnInvalidTransitionError(
                f"{self.state} 상태에서는 commit을 시작할 수 없습니다."
            )
        if expected_revision != self.revision:
            raise VoiceTurnInvalidTransitionError("제출 대상 revision이 최신값이 아닙니다.")

        answer_text = self.answer_text.strip()
        if not answer_text:
            raise VoiceTurnInvalidTransitionError("빈 답변은 제출할 수 없습니다.")
        self.speech_active = False
        self.state = "committing"
        return answer_text

    def abort_commit(self, *, question_id: str, expected_revision: int) -> None:
        """실패하거나 무효화된 제출을 취소하고 다시 답변을 수집한다.

        Args:
            question_id:
                취소할 제출의 질문 ID.

            expected_revision:
                취소할 제출의 revision.

        Raises:
            VoiceTurnQuestionMismatchError:
                현재 질문과 다른 질문 ID가 전달된 경우.

            VoiceTurnInvalidTransitionError:
                committing 상태가 아니거나 revision이 달라진 경우.
        """
        self._validate_question_id(question_id)
        self._require_state("committing")
        if expected_revision != self.revision:
            raise VoiceTurnInvalidTransitionError("취소 대상 revision이 현재값과 다릅니다.")
        self.state = "listening"

    def mark_committed(self, *, question_id: str, expected_revision: int) -> None:
        """기존 제출 경로의 성공 결과를 현재 buffer에 확정한다.

        Args:
            question_id:
                제출 완료된 답변의 질문 ID.

            expected_revision:
                제출 완료된 답변의 revision.

        Raises:
            VoiceTurnQuestionMismatchError:
                현재 질문과 다른 질문 ID가 전달된 경우.

            VoiceTurnInvalidTransitionError:
                committing 상태가 아니거나 revision이 달라진 경우.
        """
        self._validate_question_id(question_id)
        self._require_state("committing")
        if expected_revision != self.revision:
            raise VoiceTurnInvalidTransitionError("제출 완료 revision이 현재값과 다릅니다.")
        self.speech_active = False
        self.committed_revision = expected_revision
        self.state = "committed"

    def _validate_question_id(self, question_id: str) -> None:
        """이벤트 질문 ID가 현재 buffer 질문과 같은지 확인한다.

        Args:
            question_id:
                검증할 질문 ID.

        Raises:
            VoiceTurnQuestionMismatchError:
                현재 질문 ID와 다른 경우.
        """
        if question_id != self.question_id:
            raise VoiceTurnQuestionMismatchError(
                f"현재 질문과 다른 음성 이벤트입니다: {question_id}"
            )

    def _ensure_not_committed(self) -> None:
        """현재 buffer가 이미 제출된 상태가 아닌지 확인한다.

        Raises:
            VoiceTurnAlreadyCommittedError:
                state가 committed인 경우.
        """
        if self.state == "committed":
            raise VoiceTurnAlreadyCommittedError("이미 제출된 음성 답변입니다.")

    def _require_state(self, expected_state: VoiceTurnState) -> None:
        """현재 상태가 요청한 전이의 시작 상태인지 확인한다.

        Args:
            expected_state:
                상태 전이가 요구하는 현재 상태.

        Raises:
            VoiceTurnInvalidTransitionError:
                실제 상태가 expected_state와 다른 경우.
        """
        if self.state != expected_state:
            raise VoiceTurnInvalidTransitionError(
                f"{self.state} 상태에서는 이 동작을 수행할 수 없습니다."
            )

    def _validate_actionable_decision(
        self,
        *,
        expected_revision: int,
        recommended_action: Literal["auto_submit", "ask_confirmation"],
    ) -> None:
        """현재 최신 판단이 요청한 상태 전이와 일치하는지 확인한다.

        Args:
            expected_revision:
                상태 전이에 사용할 전사문 revision.

            recommended_action:
                상태 전이가 요구하는 최신 판단의 권장 동작.

        Raises:
            VoiceTurnInvalidTransitionError:
                revision, 최신 판단 또는 권장 동작이 현재 상태와 맞지 않는 경우.
        """
        if expected_revision != self.revision:
            raise VoiceTurnInvalidTransitionError("상태 전이 revision이 최신값이 아닙니다.")
        if (
            self.latest_decision_revision != expected_revision
            or self.latest_decision is None
            or self.latest_decision.recommended_action != recommended_action
        ):
            raise VoiceTurnInvalidTransitionError(
                "현재 revision에 상태 전이를 허용하는 최신 판단이 없습니다."
            )
