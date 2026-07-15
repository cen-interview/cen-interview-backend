"""최신 음성 전사 snapshot만 완료 판단하는 비동기 worker."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from time import monotonic

from interview.config import settings
from interview.interviewer.turn_completion.buffer import VoiceTurnBuffer
from interview.interviewer.turn_completion.judge import TurnCompletionJudge
from interview.interviewer.turn_completion.models import (
    TurnCompletionResult,
    TurnCompletionSnapshot,
)


TurnCompletionResultCallback = Callable[[TurnCompletionResult], Awaitable[None]]
"""최신 완료 판단이 buffer에 반영된 뒤 호출할 비동기 callback 타입."""


class LatestWinsTurnCompletionWorker:
    """세션의 현재 질문에서 최신 snapshot만 순차적으로 판단한다.

    동시에 하나의 LLM 판단만 실행한다. 판단 중 새 요청이 들어오면 큐에 모두
    쌓지 않고 가장 높은 revision의 snapshot 하나만 pending으로 유지한다.
    판단 결과는 현재 buffer의 질문과 revision이 그대로일 때만 기록한다.

    이 객체는 task와 lock을 보유하는 런타임 객체이므로 Pydantic 모델이나 외부
    저장소에 직렬화하지 않는다.

    Attributes:
        _judge:
            구조화된 문맥 완료 판단을 수행하는 비동기 판단기.

        _buffer:
            worker가 담당하는 현재 질문의 실시간 음성 답변 buffer.

        _buffer_lock:
            buffer 검증과 판단 결과 기록을 직렬화하는 세션별 비동기 lock.

        _state_lock:
            pending snapshot과 runner task 같은 worker 내부 상태를 보호하는 lock.

        _pending_snapshot:
            다음에 판단할 최신 snapshot 하나. 더 높은 revision으로 교체된다.

        _runner_task:
            순차 판단 loop를 실행하는 단일 비동기 task.

        _closed:
            worker가 취소돼 더 이상 요청을 받지 않는지 여부.
    """

    def __init__(
        self,
        *,
        judge: TurnCompletionJudge,
        buffer: VoiceTurnBuffer,
        buffer_lock: asyncio.Lock,
        min_text_length: int | None = None,
        max_calls_per_second: float | None = None,
        on_result: TurnCompletionResultCallback | None = None,
    ) -> None:
        """최신 snapshot 우선 완료 판단 worker를 생성한다.

        Args:
            judge:
                각 snapshot의 완료 여부를 판단할 TurnCompletionJudge.

            buffer:
                worker가 담당할 현재 세션과 질문의 VoiceTurnBuffer.

            buffer_lock:
                buffer의 읽기와 판단 결과 기록을 보호할 세션별 asyncio.Lock.

            min_text_length:
                첫 판단과 이후 의미 있는 텍스트 증가에 사용할 최소 문자 수.
                공백을 제외해 계산하며, 없으면 애플리케이션 설정을 사용한다.

            max_calls_per_second:
                세션 하나에서 허용할 초당 최대 LLM 판단 시작 횟수. 없으면
                애플리케이션 설정을 사용한다.

            on_result:
                최신 판단이 buffer에 실제 반영된 뒤 호출할 선택적 비동기
                callback. 상태 정책 적용은 이후 coordinator가 담당한다.

        Raises:
            ValueError:
                최소 텍스트 길이 또는 초당 최대 호출 수가 0 이하인 경우.
        """
        configured_min_length = (
            settings.turn_completion_min_text_length
            if min_text_length is None
            else min_text_length
        )
        configured_max_calls = (
            settings.turn_completion_max_calls_per_second
            if max_calls_per_second is None
            else max_calls_per_second
        )
        if configured_min_length <= 0:
            raise ValueError("완료 판단 최소 텍스트 길이는 0보다 커야 합니다.")
        if configured_max_calls <= 0:
            raise ValueError("완료 판단 초당 최대 호출 수는 0보다 커야 합니다.")

        self._judge = judge
        self._buffer = buffer
        self._buffer_lock = buffer_lock
        self._state_lock = asyncio.Lock()
        self._min_text_length = configured_min_length
        self._min_call_interval = 1.0 / configured_max_calls
        self._on_result = on_result
        self._pending_snapshot: TurnCompletionSnapshot | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._latest_accepted_revision = -1
        self._latest_accepted_text = ""
        self._latest_accepted_segment_final = False
        self._last_call_started_at: float | None = None
        self._closed = False

    @property
    def buffer(self) -> VoiceTurnBuffer:
        """worker가 담당하는 VoiceTurnBuffer를 반환한다."""
        return self._buffer

    @property
    def buffer_lock(self) -> asyncio.Lock:
        """worker가 buffer 갱신에 사용하는 세션별 비동기 lock을 반환한다."""
        return self._buffer_lock

    @property
    def closed(self) -> bool:
        """worker가 취소돼 새 snapshot을 받지 않는지 반환한다."""
        return self._closed

    async def submit(self, snapshot: TurnCompletionSnapshot) -> bool:
        """판단할 최신 전사 snapshot을 worker에 제출한다.

        buffer와 일치하는 listening 상태의 snapshot만 받는다. 첫 판단은 최소
        길이를 충족하면 즉시 예약하고, 이후에는 안정화된 STT 구간이거나 직전
        수락 snapshot보다 최소 길이만큼 증가한 경우에만 새 판단 cycle을
        시작한다. 실행 중이거나 rate-limit 대기 중인 runner가 있으면 작은
        변경도 더 높은 revision의 pending snapshot 하나로 교체한다.

        Args:
            snapshot:
                현재 질문과 누적 전사문 최신본을 담은 완료 판단 입력.

        Returns:
            snapshot을 최신 판단 후보로 수락했으면 True. 상태 불일치, 짧거나
            의미 없는 변경, 오래된 revision 또는 닫힌 worker이면 False.
        """
        if self._closed:
            return False

        async with self._buffer_lock:
            if not self._matches_current_buffer(snapshot):
                return False

        normalized_text = " ".join(snapshot.current_answer.split())
        compact_text = normalized_text.replace(" ", "")
        if len(compact_text) < self._min_text_length:
            return False

        async with self._state_lock:
            if self._closed or snapshot.revision <= self._latest_accepted_revision:
                return False
            snapshot_changed = (
                compact_text != self._latest_accepted_text
                or snapshot.segment_final != self._latest_accepted_segment_final
            )
            if not snapshot_changed:
                return False

            runner_active = (
                self._runner_task is not None and not self._runner_task.done()
            )
            if not runner_active and not self._is_meaningful_update(
                snapshot=snapshot,
                compact_text=compact_text,
            ):
                return False

            self._latest_accepted_revision = snapshot.revision
            self._latest_accepted_text = compact_text
            self._latest_accepted_segment_final = snapshot.segment_final
            self._pending_snapshot = snapshot
            if self._runner_task is None or self._runner_task.done():
                self._runner_task = asyncio.create_task(self._run())
        return True

    def cancel(self) -> None:
        """대기 snapshot을 제거하고 실행 중인 runner에 취소를 요청한다.

        이 메서드는 registry의 동기 질문 교체·제거 경계에서도 호출할 수 있도록
        기다리지 않는다. task 종료까지 기다려야 하는 호출부는 aclose를 사용한다.
        """
        self._closed = True
        self._pending_snapshot = None
        task = self._runner_task
        if task is not None and not task.done():
            task.cancel()

    async def aclose(self) -> None:
        """worker를 취소하고 실행 중인 runner task 종료까지 기다린다."""
        task = self._runner_task
        self.cancel()
        if task is None or task is asyncio.current_task():
            return
        with suppress(asyncio.CancelledError):
            await task

    def _matches_current_buffer(self, snapshot: TurnCompletionSnapshot) -> bool:
        """snapshot이 worker의 현재 buffer 최신 상태와 같은지 확인한다.

        Args:
            snapshot:
                worker에 제출하려는 완료 판단 입력.

        Returns:
            세션, 질문, revision, 전사문과 발화 상태가 모두 현재 buffer와 같고
            buffer가 listening 상태이면 True.
        """
        return (
            self._buffer.state == "listening"
            and snapshot.session_id == self._buffer.session_id
            and snapshot.question_id == self._buffer.question_id
            and snapshot.revision == self._buffer.revision
            and snapshot.current_answer == self._buffer.answer_text
            and snapshot.speech_active == self._buffer.speech_active
            and snapshot.segment_final == self._buffer.segment_final
        )

    def _is_meaningful_update(
        self,
        *,
        snapshot: TurnCompletionSnapshot,
        compact_text: str,
    ) -> bool:
        """snapshot이 새 LLM 판단을 예약할 만큼 의미 있게 변했는지 확인한다.

        Args:
            snapshot:
                판단 후보인 최신 전사 snapshot.

            compact_text:
                공백을 제거한 현재 누적 답변 문자열.

        Returns:
            첫 유효 snapshot, 안정화된 STT 구간 또는 직전 수락 이후 최소 길이
            이상의 텍스트 증가이면 True.
        """
        if self._latest_accepted_revision < 0:
            return True
        if snapshot.segment_final:
            return (
                compact_text != self._latest_accepted_text
                or not self._latest_accepted_segment_final
            )
        text_growth = len(compact_text) - len(self._latest_accepted_text)
        return text_growth >= self._min_text_length

    async def _run(self) -> None:
        """호출 간격을 지키며 pending 최신 snapshot을 순차 판단한다."""
        current_task = asyncio.current_task()
        try:
            while not self._closed:
                async with self._state_lock:
                    if self._pending_snapshot is None:
                        return
                    delay = self._remaining_call_delay()

                if delay > 0:
                    await asyncio.sleep(delay)
                    continue

                async with self._state_lock:
                    if self._closed or self._pending_snapshot is None:
                        continue
                    snapshot = self._pending_snapshot
                    self._pending_snapshot = None
                    self._last_call_started_at = monotonic()

                try:
                    result = await self._judge.judge(snapshot)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    continue

                applied = await self._apply_latest_result(result)
                if applied and self._on_result is not None:
                    try:
                        await self._on_result(result)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        continue
        finally:
            async with self._state_lock:
                if self._runner_task is current_task:
                    self._runner_task = None
                    if not self._closed and self._pending_snapshot is not None:
                        self._runner_task = asyncio.create_task(self._run())

    def _remaining_call_delay(self) -> float:
        """다음 LLM 판단을 시작하기까지 남은 rate-limit 시간을 반환한다.

        Returns:
            첫 호출이면 0. 이후 호출이면 최소 호출 간격에서 이미 지난 시간을
            뺀 0 이상의 초 단위 값.
        """
        if self._last_call_started_at is None:
            return 0.0
        elapsed = monotonic() - self._last_call_started_at
        return max(0.0, self._min_call_interval - elapsed)

    async def _apply_latest_result(self, result: TurnCompletionResult) -> bool:
        """현재 buffer와 질문·revision이 같은 판단 결과만 기록한다.

        Args:
            result:
                LLM 판단 입력의 질문 ID와 revision에 연결된 완료 판단 결과.

        Returns:
            최신 판단을 buffer에 기록했으면 True. worker가 닫혔거나 질문,
            revision 또는 상태가 달라 폐기했으면 False.
        """
        if self._closed:
            return False
        async with self._buffer_lock:
            if self._closed:
                return False
            if result.question_id != self._buffer.question_id:
                return False
            if result.revision != self._buffer.revision:
                return False
            return self._buffer.record_decision(result)
