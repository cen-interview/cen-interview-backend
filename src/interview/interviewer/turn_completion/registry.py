"""세션별 실시간 음성 턴 buffer를 보관하는 인메모리 registry."""

from asyncio import Lock as AsyncLock
from dataclasses import dataclass
from threading import Lock

from interview.interviewer.turn_completion.buffer import VoiceTurnBuffer
from interview.interviewer.turn_completion.worker import (
    LatestWinsTurnCompletionWorker,
)


@dataclass(slots=True)
class VoiceTurnRegistryEntry:
    """세션의 현재 음성 턴 상태와 비동기 갱신 lock을 묶는다.

    Attributes:
        buffer:
            현재 질문의 직렬화 가능한 실시간 음성 답변 상태.

        lock:
            같은 세션에서 들어오는 전사, 발화와 판단 결과 갱신을 순서대로
            처리하기 위한 비동기 lock.

        worker:
            현재 질문의 최신 snapshot 완료 판단을 실행하는 선택적 런타임
            worker. JSON 직렬화 대상이 아니며 질문 교체나 세션 제거 시 취소한다.
    """

    buffer: VoiceTurnBuffer
    lock: AsyncLock
    worker: LatestWinsTurnCompletionWorker | None = None


class VoiceTurnRegistry:
    """프로세스 메모리에서 세션별 활성 음성 턴을 관리한다.

    기존 Interviewer SessionRegistry는 확정된 면접 그래프와 의존성을 관리하고,
    이 registry는 언제든 폐기될 수 있는 부분 전사문과 음성 제어 상태만
    관리한다. registry 자체 lock은 항목 dict의 짧은 등록·조회·교체 작업만
    보호하고 실제 비동기 상태 갱신은 각 항목의 AsyncLock으로 보호한다.

    서버 재시작이나 여러 프로세스 사이에는 상태가 유지·공유되지 않는다.
    운영 환경에서 복구가 필요하면 직렬화 가능한 VoiceTurnBuffer만 Redis 같은
    외부 저장소로 옮기고 lock과 실행 중 task는 프로세스에 남겨야 한다.
    """

    def __init__(self) -> None:
        """빈 세션별 음성 턴 저장소와 registry 보호용 lock을 생성한다."""
        self._entries: dict[str, VoiceTurnRegistryEntry] = {}
        self._lock = Lock()

    def open_turn(
        self,
        *,
        session_id: str,
        question_id: str,
    ) -> VoiceTurnRegistryEntry:
        """세션의 현재 질문에 해당하는 음성 턴을 열거나 반환한다.

        같은 세션과 질문이 이미 등록돼 있으면 reconnect 상황으로 보고 기존
        buffer를 유지한다. 질문이 달라졌으면 이전 확인·판단 상태를 폐기하고
        revision 0의 새 buffer와 새 비동기 lock으로 교체한다.

        Args:
            session_id:
                음성 턴이 속한 면접 세션 ID.

            question_id:
                현재 지원자가 답변할 질문 ID.

        Returns:
            현재 질문의 VoiceTurnBuffer와 세션별 AsyncLock을 담은 registry 항목.

        Raises:
            ValueError:
                session_id 또는 question_id가 비어 있는 경우.
        """
        normalized_session_id = session_id.strip()
        normalized_question_id = question_id.strip()
        if not normalized_session_id:
            raise ValueError("음성 턴 session_id는 비어 있을 수 없습니다.")
        if not normalized_question_id:
            raise ValueError("음성 턴 question_id는 비어 있을 수 없습니다.")

        with self._lock:
            current_entry = self._entries.get(normalized_session_id)
            if (
                current_entry is not None
                and current_entry.buffer.question_id == normalized_question_id
            ):
                return current_entry

            if current_entry is not None and current_entry.worker is not None:
                current_entry.worker.cancel()

            entry = VoiceTurnRegistryEntry(
                buffer=VoiceTurnBuffer(
                    session_id=normalized_session_id,
                    question_id=normalized_question_id,
                ),
                lock=AsyncLock(),
            )
            self._entries[normalized_session_id] = entry
            return entry

    def replace_question(
        self,
        *,
        session_id: str,
        question_id: str,
    ) -> VoiceTurnRegistryEntry:
        """세션의 기존 음성 턴을 새 질문의 초기 buffer로 교체한다.

        같은 질문 ID를 전달해도 명시적인 초기화 요청으로 취급해 새 buffer를
        만든다. 실행 중 판단 task의 취소는 다음 latest-wins worker 단계에서
        담당하며, 이 메서드는 이전 buffer 참조를 registry에서 제거한다.

        Args:
            session_id:
                질문을 교체할 면접 세션 ID.

            question_id:
                새로 시작할 질문 ID.

        Returns:
            revision 0과 listening 상태로 초기화된 새 registry 항목.

        Raises:
            ValueError:
                session_id 또는 question_id가 비어 있는 경우.
        """
        normalized_session_id = session_id.strip()
        normalized_question_id = question_id.strip()
        if not normalized_session_id:
            raise ValueError("음성 턴 session_id는 비어 있을 수 없습니다.")
        if not normalized_question_id:
            raise ValueError("음성 턴 question_id는 비어 있을 수 없습니다.")

        entry = VoiceTurnRegistryEntry(
            buffer=VoiceTurnBuffer(
                session_id=normalized_session_id,
                question_id=normalized_question_id,
            ),
            lock=AsyncLock(),
        )
        with self._lock:
            current_entry = self._entries.get(normalized_session_id)
            if current_entry is not None and current_entry.worker is not None:
                current_entry.worker.cancel()
            self._entries[normalized_session_id] = entry
        return entry

    def attach_worker(
        self,
        *,
        session_id: str,
        worker: LatestWinsTurnCompletionWorker,
    ) -> VoiceTurnRegistryEntry:
        """현재 세션과 질문의 registry 항목에 완료 판단 worker를 연결한다.

        기존 worker가 있으면 먼저 취소해 세션 하나에서 runner가 중복 실행되지
        않게 한다. 같은 질문 ID라도 교체된 이전 buffer에 묶인 worker는 연결할
        수 없도록 buffer 객체 identity를 확인한다.

        Args:
            session_id:
                worker를 연결할 면접 세션 ID.

            worker:
                현재 registry buffer와 lock을 사용하도록 생성한 latest-wins
                완료 판단 worker.

        Returns:
            worker가 연결된 현재 VoiceTurnRegistryEntry.

        Raises:
            KeyError:
                등록되지 않은 session_id인 경우.

            ValueError:
                닫힌 worker이거나 현재 registry buffer가 아닌 객체에 연결된
                worker인 경우.
        """
        with self._lock:
            entry = self._entries[session_id]
            if worker.closed:
                raise ValueError("취소된 완료 판단 worker는 연결할 수 없습니다.")
            if worker.buffer is not entry.buffer:
                raise ValueError("현재 음성 턴 buffer에 연결된 worker가 아닙니다.")
            if worker.buffer_lock is not entry.lock:
                raise ValueError("현재 음성 턴 lock을 사용하는 worker가 아닙니다.")
            if entry.worker is worker:
                return entry
            if entry.worker is not None:
                entry.worker.cancel()
            entry.worker = worker
            return entry

    def detach_worker(
        self,
        *,
        session_id: str,
        worker: LatestWinsTurnCompletionWorker,
    ) -> bool:
        """현재 연결이 소유한 worker만 registry 항목에서 분리한다.

        reconnect로 새 worker가 이미 연결된 경우 이전 WebSocket의 종료 처리가
        새 worker를 제거하지 않도록 객체 identity를 비교한다. buffer는
        재연결 복구를 위해 registry에 그대로 유지한다.

        Args:
            session_id:
                worker를 분리할 면접 세션 ID.

            worker:
                종료 중인 WebSocket coordinator가 소유한 worker.

        Returns:
            현재 registry worker와 일치해 분리했으면 True. 세션이 없거나 이미
            다른 worker로 교체됐으면 False.
        """
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None or entry.worker is not worker:
                return False
            entry.worker.cancel()
            entry.worker = None
            return True

    def get(self, session_id: str) -> VoiceTurnRegistryEntry:
        """등록된 세션의 현재 음성 턴 항목을 반환한다.

        Args:
            session_id:
                조회할 면접 세션 ID.

        Returns:
            현재 질문의 buffer와 비동기 lock을 담은 registry 항목.

        Raises:
            KeyError:
                등록되지 않은 session_id인 경우.
        """
        with self._lock:
            return self._entries[session_id]

    def remove(self, session_id: str) -> VoiceTurnRegistryEntry | None:
        """세션의 음성 턴 상태를 registry에서 제거한다.

        일시적인 WebSocket 단절에서 즉시 호출하지 않고, 면접 종료나 명시적인
        턴 폐기 또는 reconnect 유예 만료 시 coordinator가 호출한다.

        Args:
            session_id:
                제거할 면접 세션 ID.

        Returns:
            제거한 항목. 등록된 항목이 없으면 None.
        """
        with self._lock:
            entry = self._entries.pop(session_id, None)
            if entry is not None and entry.worker is not None:
                entry.worker.cancel()
            return entry


_voice_turn_registry = VoiceTurnRegistry()


def get_voice_turn_registry() -> VoiceTurnRegistry:
    """애플리케이션 프로세스의 공용 음성 턴 registry를 반환한다.

    Returns:
        세션별 현재 VoiceTurnBuffer를 보관하는 인메모리 registry.
    """
    return _voice_turn_registry
