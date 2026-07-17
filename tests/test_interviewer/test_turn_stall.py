"""답변 정체(stall) timeout의 확인 질문 강제 시작과 듣기 중단을 검증한다."""

import asyncio
from dataclasses import dataclass, field

import pytest

from interview.interviewer.turn_completion.buffer import (
    VoiceTurnBuffer,
    VoiceTurnInvalidTransitionError,
)
from interview.interviewer.turn_completion.coordinator import (
    CONFIRMATION_PROMPT_TEXT,
    VoiceTurnCoordinator,
)
from interview.interviewer.turn_completion.models import (
    ConfirmationIntentDecision,
    TurnCompletionDecision,
    TurnCompletionResult,
)
from interview.interviewer.turn_completion.registry import VoiceTurnRegistryEntry


def _decision(
    *,
    recommended_action: str = "keep_listening",
    semantic_state: str = "incomplete",
    confidence: float = 0.95,
) -> TurnCompletionDecision:
    return TurnCompletionDecision(
        semantic_state=semantic_state,
        linguistically_closed=semantic_state != "incomplete",
        question_satisfied=semantic_state != "incomplete",
        continuation_expected="medium",
        explicit_completion=False,
        recommended_action=recommended_action,
        confidence=confidence,
        reason_code="hesitation" if semantic_state != "complete" else "complete_thought",
    )


def _buffer_with_answer(text: str = "스프링 시큐리티가 뭔가요") -> VoiceTurnBuffer:
    buffer = VoiceTurnBuffer(session_id="s1", question_id="q1")
    buffer.update_transcript(
        question_id="q1",
        revision=1,
        text=text,
        speech_active=False,
        segment_final=True,
    )
    return buffer


class TestMarkCutoffComplete:
    def test_listening_상태에서_완료_후보로_전환한다(self):
        buffer = _buffer_with_answer()

        cancelled = buffer.mark_cutoff_complete(question_id="q1", expected_revision=1)

        assert cancelled is None
        assert buffer.state == "complete_candidate"
        assert buffer.pending_completion_reason == "listening_cutoff"

    def test_confirming_end_상태에서_활성_확인을_취소하고_전환한다(self):
        buffer = _buffer_with_answer()
        buffer.record_decision(
            TurnCompletionResult(
                question_id="q1",
                revision=1,
                decision=_decision(
                    recommended_action="ask_confirmation",
                    semantic_state="ambiguous",
                ),
            )
        )
        buffer.mark_confirmation_pending(expected_revision=1, max_confirmations=1)
        buffer.begin_confirmation(
            confirmation_id="c1",
            expected_revision=1,
            max_confirmations=1,
        )

        cancelled = buffer.mark_cutoff_complete(question_id="q1", expected_revision=1)

        assert cancelled == "c1"
        assert buffer.active_confirmation_id is None
        assert buffer.state == "complete_candidate"
        assert buffer.pending_completion_reason == "listening_cutoff"

    def test_발화_중에는_중단할_수_없다(self):
        buffer = VoiceTurnBuffer(session_id="s1", question_id="q1")
        buffer.update_transcript(
            question_id="q1",
            revision=1,
            text="답변 중입니다",
            speech_active=True,
            segment_final=False,
        )

        with pytest.raises(VoiceTurnInvalidTransitionError):
            buffer.mark_cutoff_complete(question_id="q1", expected_revision=1)

    def test_빈_답변은_중단_대상이_아니다(self):
        buffer = VoiceTurnBuffer(session_id="s1", question_id="q1")

        with pytest.raises(VoiceTurnInvalidTransitionError):
            buffer.mark_cutoff_complete(question_id="q1", expected_revision=0)

    def test_오래된_revision은_거절한다(self):
        buffer = _buffer_with_answer()

        with pytest.raises(VoiceTurnInvalidTransitionError):
            buffer.mark_cutoff_complete(question_id="q1", expected_revision=0)

    def test_mark_complete_candidate는_cutoff_사유를_허용하지_않는다(self):
        buffer = _buffer_with_answer()
        buffer.record_decision(
            TurnCompletionResult(
                question_id="q1",
                revision=1,
                decision=_decision(
                    recommended_action="auto_submit",
                    semantic_state="complete",
                ),
            )
        )

        with pytest.raises(VoiceTurnInvalidTransitionError):
            buffer.mark_complete_candidate(
                expected_revision=1,
                completion_reason="listening_cutoff",
            )


@dataclass
class _FakeQuestion:
    question_id: str = "q1"
    text: str = "Spring Security는 무엇이고, 왜 사용하나요?"
    kind: str = "main"
    topic: str = "spring security"


@dataclass
class _FakeSessionState:
    finished: bool = False
    mode: str = "voice"
    current_question: _FakeQuestion = field(default_factory=_FakeQuestion)
    transcript: list = field(default_factory=list)


class _FakeInterviewSession:
    def __init__(self) -> None:
        self.state = _FakeSessionState()

    def get_state(self) -> _FakeSessionState:
        return self.state


class _StubJudge:
    """항상 같은 keep_listening 판단을 돌려주는 판단기."""

    def __init__(self, decision: TurnCompletionDecision | None = None) -> None:
        self._decision = decision or _decision()

    async def judge(self, snapshot) -> TurnCompletionResult:
        return TurnCompletionResult(
            question_id=snapshot.question_id,
            revision=snapshot.revision,
            decision=self._decision,
        )


class _StubConfirmationClassifier:
    async def classify(self, text: str) -> ConfirmationIntentDecision:
        return ConfirmationIntentDecision(intent="unknown", confidence=0.5)


async def _wait_for(predicate, *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("기대한 상태에 도달하지 못했습니다.")


def _build_coordinator(
    entry: VoiceTurnRegistryEntry,
    *,
    events: dict,
    stall_timeout_seconds: float = 0.08,
) -> VoiceTurnCoordinator:
    events.setdefault("confirmations", [])
    events.setdefault("cancelled", [])
    events.setdefault("commits", [])
    events.setdefault("committed", [])

    async def on_confirmation_requested(confirmation_id, question_id, revision, text):
        events["confirmations"].append(
            {
                "confirmation_id": confirmation_id,
                "question_id": question_id,
                "revision": revision,
                "text": text,
            }
        )

    async def on_confirmation_cancelled(confirmation_id, question_id, reason):
        events["cancelled"].append(
            {
                "confirmation_id": confirmation_id,
                "question_id": question_id,
                "reason": reason,
            }
        )

    async def on_commit_answer(request):
        events["commits"].append(request)
        return {"finished": False}

    async def on_answer_committed(result):
        events["committed"].append(result)

    return VoiceTurnCoordinator(
        session=_FakeInterviewSession(),
        entry=entry,
        judge=_StubJudge(),
        confirmation_classifier=_StubConfirmationClassifier(),
        on_confirmation_requested=on_confirmation_requested,
        on_confirmation_cancelled=on_confirmation_cancelled,
        on_commit_answer=on_commit_answer,
        on_answer_committed=on_answer_committed,
        confirmation_pause_seconds=0.01,
        commit_grace_milliseconds=10,
        stall_timeout_seconds=stall_timeout_seconds,
        max_confirmations=1,
    )


class TestStallTimeout:
    def test_keep_listening_정체_시_확인_질문을_강제_시작한다(self):
        async def scenario() -> None:
            entry = VoiceTurnRegistryEntry(
                buffer=VoiceTurnBuffer(session_id="s1", question_id="q1"),
                lock=asyncio.Lock(),
            )
            events: dict = {}
            coordinator = _build_coordinator(entry, events=events)
            try:
                await coordinator.handle_transcript_updated(
                    question_id="q1",
                    revision=1,
                    text="스프링 시큐리티가 뭔가요 뭔가요",
                    speech_active=False,
                    segment_final=True,
                )

                await _wait_for(lambda: entry.buffer.state == "confirming_end")
                assert events["confirmations"]
                assert (
                    events["confirmations"][0]["text"] == CONFIRMATION_PROMPT_TEXT
                )
            finally:
                await coordinator.aclose()

        asyncio.run(scenario())

    def test_발화_flag가_stale_true여도_정체를_감지한다(self):
        """관측된 장애 재현: speech_active=true로 고정된 채 이벤트가 끊긴 경우."""

        async def scenario() -> None:
            entry = VoiceTurnRegistryEntry(
                buffer=VoiceTurnBuffer(session_id="s1", question_id="q1"),
                lock=asyncio.Lock(),
            )
            events: dict = {}
            coordinator = _build_coordinator(entry, events=events)
            try:
                await coordinator.handle_transcript_updated(
                    question_id="q1",
                    revision=1,
                    text="스프링 시큐리티가 뭔가요 뭔가요",
                    speech_active=True,
                    segment_final=True,
                )

                await _wait_for(lambda: entry.buffer.state == "confirming_end")
                assert not entry.buffer.speech_active
                assert events["confirmations"]
            finally:
                await coordinator.aclose()

        asyncio.run(scenario())

    def test_확인_질문에_응답이_없으면_듣기를_중단하고_제출한다(self):
        async def scenario() -> None:
            entry = VoiceTurnRegistryEntry(
                buffer=VoiceTurnBuffer(session_id="s1", question_id="q1"),
                lock=asyncio.Lock(),
            )
            events: dict = {}
            coordinator = _build_coordinator(entry, events=events)
            try:
                await coordinator.handle_transcript_updated(
                    question_id="q1",
                    revision=1,
                    text="스프링 시큐리티가 뭔가요 뭔가요",
                    speech_active=False,
                    segment_final=True,
                )

                await _wait_for(lambda: entry.buffer.state == "confirming_end")
                confirmation_id = events["confirmations"][0]["confirmation_id"]

                await _wait_for(lambda: entry.buffer.state == "committed")
                assert events["commits"]
                assert events["commits"][0].completion_reason == "listening_cutoff"
                assert events["cancelled"]
                assert events["cancelled"][0]["confirmation_id"] == confirmation_id
                assert events["cancelled"][0]["reason"] == "listening_cutoff"
            finally:
                await coordinator.aclose()

        asyncio.run(scenario())

    def test_새_전사가_도착하면_정체_감시를_취소한다(self):
        async def scenario() -> None:
            entry = VoiceTurnRegistryEntry(
                buffer=VoiceTurnBuffer(session_id="s1", question_id="q1"),
                lock=asyncio.Lock(),
            )
            events: dict = {}
            coordinator = _build_coordinator(
                entry,
                events=events,
                stall_timeout_seconds=0.2,
            )
            try:
                await coordinator.handle_transcript_updated(
                    question_id="q1",
                    revision=1,
                    text="스프링 시큐리티가 뭔가요 뭔가요",
                    speech_active=False,
                    segment_final=True,
                )
                await asyncio.sleep(0.05)
                await coordinator.handle_transcript_updated(
                    question_id="q1",
                    revision=2,
                    text="스프링 시큐리티가 뭔가요 뭔가요 추가",
                    speech_active=True,
                    segment_final=False,
                )

                await asyncio.sleep(0.3)
                assert entry.buffer.state == "listening"
                assert not events["confirmations"]
                assert not events["commits"]
            finally:
                await coordinator.aclose()

        asyncio.run(scenario())
