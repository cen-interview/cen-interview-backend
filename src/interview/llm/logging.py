"""LLM 호출 결과를 서버 로그에 일관된 형식으로 출력한다.

생성 결과는 기본적으로 출력하지만, 프롬프트와 사용자 답변 같은 입력값은
민감한 내용을 포함할 수 있으므로 설정에서 명시적으로 활성화한 경우에만
출력한다.
"""

import json
import logging
from enum import Enum
from typing import Any

from interview.config import settings


_logger = logging.getLogger("uvicorn.error")


def _to_json_value(value: Any) -> Any:
    """로그 값을 JSON으로 직렬화 가능한 형태로 변환한다.

    Args:
        value:
            Pydantic 모델, Enum, dict, list 또는 일반 파이썬 값.

    Returns:
        json.dumps가 처리할 수 있는 값. 직접 변환할 수 없는 객체는 문자열로
        바꾼다.
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


def _render_payload(payload: dict[str, Any]) -> str:
    """로그 payload를 한글이 유지되는 JSON 문자열로 만든다.

    Args:
        payload:
            로그에 출력할 구조화 데이터.

    Returns:
        설정된 최대 길이를 넘지 않도록 잘라낸 JSON 문자열.
    """
    rendered = json.dumps(
        _to_json_value(payload),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    max_length = max(settings.llm_log_max_length, 0)
    if max_length and len(rendered) > max_length:
        omitted = len(rendered) - max_length
        return f"{rendered[:max_length]}\n... ({omitted}자 생략)"
    return rendered


def log_llm_output(
    event: str,
    output: Any,
    *,
    metadata: dict[str, Any] | None = None,
    input_data: Any = None,
    status: str = "success",
) -> None:
    """LLM 생성 결과를 서버 로그에 출력한다.

    API 서버에서 사용하는 uvicorn error logger를 이용하므로 PyCharm 또는
    uvicorn 실행 터미널에서 기존 서버 로그와 함께 확인할 수 있다. 입력값은
    ``LLM_LOG_INCLUDE_INPUT=true``일 때만 포함한다.

    Args:
        event:
            로그를 구분하는 이벤트 이름. 예: ANSWER_ASSESSMENT.

        output:
            LLM이 반환했거나 폴백으로 선택된 결과.

        metadata:
            question_id, topic, kind처럼 결과를 찾는 데 필요한 보조 정보.

        input_data:
            프롬프트나 사용자 답변 등 선택적인 LLM 입력값.

        status:
            success, fallback, template처럼 생성 경로를 나타내는 상태.
    """
    if not settings.llm_log_enabled:
        return

    payload: dict[str, Any] = {
        "metadata": metadata or {},
        "output": output,
    }
    if settings.llm_log_include_input and input_data is not None:
        payload["input"] = input_data

    _logger.info(
        "[LLM][%s][%s]\n%s",
        event,
        status.upper(),
        _render_payload(payload),
    )


def log_llm_error(
    event: str,
    error: Exception,
    *,
    metadata: dict[str, Any] | None = None,
    fallback: Any = None,
    input_data: Any = None,
) -> None:
    """LLM 호출 실패와 선택된 폴백 결과를 서버 로그에 출력한다.

    Args:
        event:
            실패한 LLM 작업을 구분하는 이벤트 이름.

        error:
            LLM 호출 중 발생한 예외.

        metadata:
            question_id, topic, kind 같은 보조 정보.

        fallback:
            호출부가 대신 사용할 템플릿 또는 임시 결과.

        input_data:
            설정으로 허용했을 때만 출력할 LLM 입력값.
    """
    if not settings.llm_log_enabled:
        return

    payload: dict[str, Any] = {
        "metadata": metadata or {},
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "fallback": fallback,
    }
    if settings.llm_log_include_input and input_data is not None:
        payload["input"] = input_data

    _logger.warning(
        "[LLM][%s][FALLBACK]\n%s",
        event,
        _render_payload(payload),
    )
