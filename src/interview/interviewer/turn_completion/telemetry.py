"""문맥 기반 음성 턴의 구조화된 관찰 로그를 기록한다.

답변 완료 정책을 조정하는 데 필요한 상태, revision, latency와 사유 코드를
일관된 JSON 형태로 남긴다. 답변 원문은 민감한 정보일 수 있으므로 기존
``llm_log_include_input`` 설정이 명시적으로 활성화된 경우에만 포함한다.
"""

import json
import logging
from enum import Enum
from time import monotonic
from typing import Any

from interview.config import settings


_logger = logging.getLogger("uvicorn.error")


def monotonic_time() -> float:
    """latency 측정에 사용할 단조 증가 시각을 반환한다.

    Returns:
        시스템 시각 변경의 영향을 받지 않는 초 단위 monotonic 값.
    """
    return monotonic()


def elapsed_milliseconds(started_at: float) -> int:
    """시작 시각부터 현재까지의 경과 시간을 밀리초로 변환한다.

    Args:
        started_at:
            ``monotonic_time()``으로 측정한 작업 시작 시각.

    Returns:
        음수가 되지 않도록 보정한 정수 밀리초.
    """
    return max(0, round((monotonic() - started_at) * 1000))


def log_voice_turn_event(
    event: str,
    *,
    session_id: str,
    question_id: str | None = None,
    revision: int | None = None,
    answer_text: str | None = None,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """음성 턴 이벤트를 공통 필드의 구조화 로그로 기록한다.

    Args:
        event:
            ``voice_turn.judge.completed``처럼 안정적인 이벤트 이름.

        session_id:
            관찰 대상 면접 세션 ID.

        question_id:
            관찰 대상 질문 ID. 연결 초기처럼 알 수 없으면 None.

        revision:
            관찰 대상 전사문 revision. 없으면 None.

        answer_text:
            길이 계산에 사용할 선택적 답변 원문. 실제 원문은
            ``llm_log_include_input``이 True일 때만 로그에 포함한다.

        level:
            Python logging level. 기본값은 INFO.

        **fields:
            상태, 판단, latency, 완료 사유 등 이벤트별 추가 필드.
    """
    payload: dict[str, Any] = {
        "event": event,
        "session_id": session_id,
    }
    if question_id is not None:
        payload["question_id"] = question_id
    if revision is not None:
        payload["revision"] = revision
    if answer_text is not None:
        payload["answer_length"] = len(answer_text)
        if settings.llm_log_include_input:
            payload["answer_text"] = answer_text
    payload.update(
        {
            key: _to_json_value(value)
            for key, value in fields.items()
            if value is not None
        }
    )

    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    max_length = max(settings.llm_log_max_length, 0)
    if max_length and len(rendered) > max_length:
        rendered = f"{rendered[:max_length]}..."
    _logger.log(level, "[VOICE_TURN] %s", rendered)


def _to_json_value(value: Any) -> Any:
    """로그 필드를 JSON 직렬화 가능한 값으로 변환한다.

    Args:
        value:
            Pydantic 모델, Enum, collection 또는 일반 값.

    Returns:
        json.dumps가 처리할 수 있는 값. 직접 변환할 수 없으면 문자열.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
