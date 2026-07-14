"""면접관 안내 문장을 생성하고 질문 원문과 조립하는 발화 레이어."""

import json
import re
from queue import Empty, Queue
from threading import Thread
from typing import Any

from interview.interviewer.models import ComposedUtterance
from interview.interviewer.session import SessionState, Turn
from interview.interviewer.speech import utterance as utterance_templates
from interview.interviewer.speech.prompts import (
    UTTERANCE_SYSTEM_PROMPT,
    build_utterance_user_prompt,
)
from interview.interviewer.speech.reaction import (
    ReactionPolicy,
    ReactionTone,
    select_reaction_policy,
)
from interview.interviewer.workflow.runtime import _runtime_deps, _state_get
from interview.llm.logging import log_llm_error, log_llm_output
from interview.schemas.question import Question, QuestionKind
from interview.schemas.signals import AnswerQuality

_UTTERANCE_LLM_TIMEOUT_SECONDS = 3.0
_UTTERANCE_TRANSCRIPT_TURN_LIMIT = 4
_UTTERANCE_TRANSCRIPT_TEXT_LIMIT = 500
_QUESTION_ANNOUNCEMENT_EXPRESSIONS = (
    "다음 질문",
    "질문을 드리",
    "질문드리",
    "여쭤보",
    "더 확인하겠",
    "추가로 확인하겠",
    "조금 더 살펴보겠",
)
_UNCONDITIONAL_APPROVAL_EXPRESSIONS = (
    "잘 들었습니다",
    "감사합니다",
    "훌륭",
    "정확하게",
    "정확히",
    "좋은 답변",
    "잘 설명",
    "핵심을 잘",
    "충분히 설명",
)
_CORRECTIVE_EXPRESSIONS = (
    "틀렸",
    "잘못",
    "오류",
    "부족",
    "불충분",
    "맞지 않",
    "모순",
    "다시 짚",
    "재검토",
)
_QUESTION_OVERLAP_STOPWORDS = frozenset(
    {
        "경우",
        "관련",
        "내용",
        "다음",
        "답변",
        "대해",
        "대한",
        "방법",
        "말씀",
        "무엇",
        "사용",
        "설명",
        "어떤",
        "어떻게",
        "예시",
        "이유",
        "의미",
        "장점",
        "정도",
        "조금",
        "주세",
        "질문",
        "차이",
    }
)
_KOREAN_PARTICLE_SUFFIXES = (
    "으로부터",
    "에서부터",
    "에게서",
    "까지는",
    "에서는",
    "으로는",
    "이라는",
    "라고는",
    "부터",
    "까지",
    "에게",
    "에서",
    "으로",
    "라고",
    "이며",
    "에는",
    "에게",
    "처럼",
    "보다",
    "과는",
    "와는",
    "이라",
    "라고",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "와",
    "과",
    "도",
    "로",
)


def _content_tokens(text: str) -> set[str]:
    """질문과 리액션의 내용 중복 검사에 사용할 핵심 토큰을 추린다.

    대소문자와 한국어 조사의 차이 때문에 같은 용어가 다른 문자열로 보이는
    문제를 줄이기 위해 영문은 소문자로 바꾸고 흔한 조사를 제거한다. 질문의
    지시 표현처럼 내용 중복 판단에 도움이 되지 않는 일반 단어는 제외한다.

    Args:
        text:
            핵심 토큰을 추출할 질문 또는 preamble 문자열.

    Returns:
        조사와 일반 지시 표현이 제거된 중복 검사용 토큰 집합.
    """
    raw_tokens = re.findall(r"[0-9A-Za-z가-힣+#]+", text.lower())
    normalized_tokens: set[str] = set()
    for raw_token in raw_tokens:
        token = raw_token
        for suffix in _KOREAN_PARTICLE_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
                break
        if len(token) < 2 or token in _QUESTION_OVERLAP_STOPWORDS:
            continue
        normalized_tokens.add(token)
    return normalized_tokens


def _preamble_repeats_question(preamble: str, question: Question) -> bool:
    """preamble이 현재 질문의 주제나 핵심 표현을 다시 말하는지 확인한다.

    질문 전체 문장이 그대로 들어간 경우뿐 아니라 topic 또는 의미 있는 단어가
    하나라도 다시 사용된 경우를 중복으로 본다. preamble은 질문 설명이 아니라
    직전 답변에 대한 리액션만 담당하므로 보수적으로 차단하고 템플릿 폴백을
    선택한다.

    Args:
        preamble:
            LLM이 생성한 면접관 리액션.

        question:
            리액션 뒤에 원문 그대로 붙일 현재 질문.

    Returns:
        질문 내용과 중복되면 True, 중복이 없으면 False.
    """
    compact_preamble = re.sub(r"\s+", "", preamble).lower()
    compact_question = re.sub(r"\s+", "", question.text).lower()
    if compact_question and compact_question in compact_preamble:
        return True

    compact_topic = re.sub(r"\s+", "", question.topic).lower()
    if len(compact_topic) >= 2 and compact_topic in compact_preamble:
        return True

    return bool(_content_tokens(preamble) & _content_tokens(question.text))


def _preamble_announces_question(preamble: str, turn_type: str) -> bool:
    """일반 질문용 preamble이 다음 질문을 예고하는 표현인지 확인한다.

    replay나 hint처럼 상황 안내 자체가 필요한 턴은 검사하지 않는다. Strategy가
    만든 새 질문 앞에 붙는 question 턴에서만 전환·예고 문구를 거부한다.

    Args:
        preamble:
            LLM이 생성한 면접관 리액션.

        turn_type:
            greeting, question, replay 등 현재 발화 상황.

    Returns:
        불필요한 질문 예고 표현이 있으면 True, 없으면 False.
    """
    if turn_type != "question":
        return False
    return any(expression in preamble for expression in _QUESTION_ANNOUNCEMENT_EXPRESSIONS)


def _preamble_conflicts_with_reaction_tone(
    preamble: str,
    reaction_tone: ReactionTone,
    turn_type: str,
) -> bool:
    """LLM 리액션이 평가에서 정한 반응 강도와 충돌하는지 확인한다.

    cautious와 corrective 답변에서 습관적인 칭찬이 나오면 지원자가 자신의
    답변 상태를 잘못 해석할 수 있다. 반대로 positive 답변에 오류나 부족을
    암시하는 표현이 나오면 평가와 발화가 모순된다. 완전한 의미 분석 대신
    방향이 명확한 표현만 보수적으로 검사하고, 충돌하면 평가별 템플릿으로
    폴백하도록 한다. greeting이나 replay 같은 비평가 턴은 검사하지 않는다.

    Args:
        preamble:
            LLM이 생성한 면접관 리액션.

        reaction_tone:
            직전 평가 신호에서 선택한 발화 반응 강도.

        turn_type:
            현재 발화 상황. 평가 기반 리액션을 사용하는 question에서만
            충돌 검사를 수행한다.

    Returns:
        발화가 반응 강도와 명확히 충돌하면 True, 그렇지 않으면 False.
    """
    if turn_type != "question" or reaction_tone == ReactionTone.NEUTRAL:
        return False

    if reaction_tone in {ReactionTone.CAUTIOUS, ReactionTone.CORRECTIVE}:
        return any(
            expression in preamble
            for expression in _UNCONDITIONAL_APPROVAL_EXPRESSIONS
        )

    return any(expression in preamble for expression in _CORRECTIVE_EXPRESSIONS)


def _select_utterance_preamble(
    turn_type: str,
    question_kind: QuestionKind | None,
    reaction_policy: ReactionPolicy,
) -> str:
    """현재 상황과 질문 종류에 맞는 기본 안내 문장을 선택한다.

    상황을 나타내는 turn_type을 질문 종류보다 먼저 확인한다. 같은 질문을
    사용하더라도 첫 제시, 재제시, 종료 상황의 안내 문장은 달라야 하기
    때문이다. 일반 question 상황에서는 직전 평가 신호를 우선 사용하여
    충분함, 추가 깊이, 범위 확인, 오개념, 충돌 가능성을 서로 다른 문장으로
    표현한다. 평가 신호가 없을 때만 Question.kind 기반 템플릿을 사용한다.

    Args:
        turn_type:
            현재 면접관 턴의 상황. greeting, question, replay,
            pause_prompt, closing 등을 사용한다.

        question_kind:
            현재 질문의 종류. 질문이 없는 종료나 일시 정지 안내에서는
            None일 수 있다.

        reaction_policy:
            직전 평가 신호에서 변환한 반응 강도와 안전한 발화 지침.

    Returns:
        현재 발화 앞에 붙일 템플릿 기반 안내 문장.
    """
    if turn_type == "greeting":
        return utterance_templates.greeting()
    if turn_type == "replay":
        return utterance_templates.replay()
    if turn_type == "pause_prompt":
        return utterance_templates.pause_prompt()
    if turn_type == "closing":
        return utterance_templates.closing()

    if turn_type == "question":
        quality_templates = {
            AnswerQuality.SUFFICIENT: utterance_templates.sufficient,
            AnswerQuality.UNKNOWN: utterance_templates.unknown,
            AnswerQuality.BONUS_AVAILABLE: utterance_templates.bonus_available,
            AnswerQuality.CONFIRM_POSITIVE: utterance_templates.confirm_positive,
            AnswerQuality.TRAP_AVAILABLE: utterance_templates.trap_available,
            AnswerQuality.MISCONCEPTION: utterance_templates.misconception,
            AnswerQuality.CONFIRM_NEGATIVE: utterance_templates.confirm_negative,
        }
        quality_template = quality_templates.get(reaction_policy.quality)
        if quality_template is not None:
            return quality_template()

    question_templates = {
        QuestionKind.FOLLOW_UP: utterance_templates.follow_up,
        QuestionKind.CHALLENGE: utterance_templates.challenge,
        QuestionKind.CONFIRM_POSITIVE: utterance_templates.follow_up,
        QuestionKind.CONFIRM_NEGATIVE: utterance_templates.challenge,
        QuestionKind.HINT: utterance_templates.hint,
    }
    template = question_templates.get(question_kind, utterance_templates.question)
    return template()


def _format_recent_transcript(state: SessionState | dict[str, Any]) -> str:
    """LLM 프롬프트에 전달할 최근 대화 몇 턴을 문자열로 정리한다.

    전체 transcript를 전달하지 않고 마지막 네 턴만 사용한다. 한 턴의 본문도
    최대 길이를 제한하여 긴 답변 때문에 발화 생성 프롬프트가 불필요하게
    커지는 것을 방지한다.

    Args:
        state:
            transcript를 가진 현재 세션 상태.

    Returns:
        역할과 발화를 줄 단위로 정리한 최근 대화. 대화가 없으면 "없음".
    """
    transcript = _state_get(state, "transcript", []) or []
    recent_turns = transcript[-_UTTERANCE_TRANSCRIPT_TURN_LIMIT:]
    if not recent_turns:
        return "없음"

    lines: list[str] = []
    for raw_turn in recent_turns:
        turn = Turn.model_validate(raw_turn) if isinstance(raw_turn, dict) else raw_turn
        text = turn.text[:_UTTERANCE_TRANSCRIPT_TEXT_LIMIT]
        lines.append(f"- {turn.role}: {text}")
    return "\n".join(lines)


def _serialize_last_signal_for_prompt(state: SessionState | dict[str, Any]) -> str:
    """직전 평가 신호를 LLM 프롬프트에 넣을 JSON 문자열로 바꾼다.

    Args:
        state:
            last_signal을 가진 현재 세션 상태.

    Returns:
        한글을 유지한 JSON 문자열. 평가 신호가 없으면 "없음".
    """
    last_signal = _state_get(state, "last_signal")
    if last_signal is None:
        return "없음"
    if hasattr(last_signal, "model_dump"):
        last_signal = last_signal.model_dump(mode="json")
    return json.dumps(last_signal, ensure_ascii=False, default=str)


def _invoke_utterance_llm_with_timeout(
    structured_llm: Any,
    messages: list[dict[str, str]],
) -> Any:
    """구조화된 LLM 호출을 제한 시간 안에서 실행한다.

    현재 Interviewer 그래프는 동기식으로 동작하므로 daemon thread에서 LLM을
    호출하고 결과 큐를 제한 시간만 기다린다. 제한 시간을 넘긴 호출은 면접
    흐름을 막지 않으며, 호출부가 즉시 템플릿 발화로 폴백한다.

    Args:
        structured_llm:
            ComposedUtterance 구조화 출력을 반환하도록 설정된 LLM runnable.

        messages:
            system과 user 메시지로 구성된 LLM 입력.

    Returns:
        제한 시간 안에 LLM이 반환한 구조화 출력.

    Raises:
        TimeoutError:
            제한 시간 안에 결과를 받지 못한 경우.

        Exception:
            LLM 호출 중 발생한 예외를 그대로 전달한다.
    """
    result_queue: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def invoke() -> None:
        """백그라운드에서 LLM을 호출하고 성공 여부와 결과를 큐에 기록한다."""
        try:
            result_queue.put((True, structured_llm.invoke(messages)))
        except Exception as exc:
            result_queue.put((False, exc))

    Thread(target=invoke, daemon=True).start()

    try:
        succeeded, result = result_queue.get(timeout=_UTTERANCE_LLM_TIMEOUT_SECONDS)
    except Empty as exc:
        raise TimeoutError("면접관 발화 생성 제한 시간을 초과했습니다.") from exc

    if succeeded:
        return result
    if isinstance(result, Exception):
        raise result
    raise RuntimeError("면접관 발화 생성 결과를 확인할 수 없습니다.")


def _generate_llm_preamble(
    *,
    llm: Any,
    state: SessionState | dict[str, Any],
    turn_type: str,
    current_question: Question | None,
    reaction_policy: ReactionPolicy,
) -> str:
    """선택적 LLM을 사용해 현재 상황의 짧은 안내 문장을 생성한다.

    LLM에는 최근 대화 일부, 직전 평가 신호와 중복 방지용 현재 질문을 전달한다.
    구조화 출력에서 preamble만 꺼내며, 질문의 주제·핵심 표현을 반복하거나
    다음 질문을 예고하면 예외를 발생시킨다. 예외는 compose_utterance가
    리액션 템플릿 폴백으로 처리한다.

    Args:
        llm:
            InterviewDeps를 통해 주입된 LLM client.

        state:
            최근 transcript와 last_signal을 가진 현재 세션 상태.

        turn_type:
            greeting, question, replay, closing 등 현재 발화 상황.

        current_question:
            현재 Strategy 질문. 종료 안내에서는 None일 수 있다.

        reaction_policy:
            직전 평가 결과에서 선택한 반응 강도와 간접 표현 지침.

    Returns:
        질문 내용을 반복하지 않는 LLM 생성 리액션.

    Raises:
        ValueError:
            LLM 결과가 비어 있거나 질문 내용을 반복·예고한 경우.

        Exception:
            구조화 출력 설정이나 LLM 호출에 실패한 경우.
    """
    question_kind = current_question.kind.value if current_question is not None else None
    question_text = current_question.text if current_question is not None else None
    user_prompt = build_utterance_user_prompt(
        turn_type=turn_type,
        question_kind=question_kind,
        question_text=question_text,
        last_signal=_serialize_last_signal_for_prompt(state),
        reaction_tone=reaction_policy.tone.value,
        reaction_guidance=reaction_policy.guidance,
        recent_transcript=_format_recent_transcript(state),
    )
    structured_llm = llm.with_structured_output(ComposedUtterance)
    result = _invoke_utterance_llm_with_timeout(
        structured_llm,
        [
            {"role": "system", "content": UTTERANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    composed = (
        result if isinstance(result, ComposedUtterance) else ComposedUtterance.model_validate(result)
    )
    preamble = composed.preamble.strip()
    if not preamble:
        raise ValueError("LLM이 빈 면접관 안내 문장을 반환했습니다.")
    if current_question is not None and _preamble_repeats_question(preamble, current_question):
        raise ValueError("LLM 리액션이 현재 질문의 내용이나 핵심 표현을 반복했습니다.")
    if _preamble_announces_question(preamble, turn_type):
        raise ValueError("LLM 리액션이 다음 질문을 불필요하게 예고했습니다.")
    if _preamble_conflicts_with_reaction_tone(
        preamble,
        reaction_policy.tone,
        turn_type,
    ):
        raise ValueError("LLM 리액션이 직전 평가의 반응 강도와 맞지 않습니다.")

    log_llm_output(
        "INTERVIEWER_PREAMBLE",
        composed,
        metadata={
            "turn_type": turn_type,
            "question_id": current_question.question_id if current_question else None,
            "question_kind": question_kind,
            "reaction_tone": reaction_policy.tone.value,
            "answer_quality": (
                reaction_policy.quality.value if reaction_policy.quality else None
            ),
        },
        input_data={"user_prompt": user_prompt},
    )
    return preamble


def _build_utterance_queue(
    preamble: str,
    question_text: str | None,
) -> list[str]:
    """TTS가 순서대로 재생할 발화 큐를 만든다.

    안내 또는 리액션 문장과 질문 원문을 서로 다른 큐 항목으로 유지한다.
    프론트는 큐를 앞에서부터 하나씩 재생하고, 마지막 항목의 재생 완료
    콜백에서 마이크를 시작할 수 있다. 질문이 없는 종료·일시정지 턴에는
    안내 문장만 담고, 비어 있는 문자열은 큐에서 제외한다.

    Args:
        preamble:
            질문 앞에 붙는 안내 또는 리액션 문장.

        question_text:
            Strategy가 만든 질문 원문. 질문이 없는 턴이면 None.

    Returns:
        TTS가 재생할 순서대로 정리된 발화 문자열 목록.
    """
    utterance_queue = [preamble.strip()]
    if question_text is not None:
        utterance_queue.append(question_text.strip())
    return [utterance for utterance in utterance_queue if utterance]


def compose_utterance(
    state: SessionState | dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """현재 상황의 안내 문장과 질문 본문을 면접관 발화로 조립한다.

    Strategy가 만든 Question.text는 수정하지 않고 짧은 preamble 앞에 그대로
    붙인다. InterviewDeps에 LLM이 있으면 구조화된 preamble 생성을 시도하고,
    LLM이 없거나 호출이 실패하거나 제한 시간을 넘기면 기본 템플릿을 사용한다.
    조립한 전체 문장은 last_utterance에 저장한다. TTS용 utterance_queue에는
    안내 또는 리액션 문장과 질문 원문을 별도 항목으로 담아 프론트가 순서대로
    재생할 수 있게 한다. 동일한 전체 문장을 interviewer Turn으로 transcript에
    추가한다. closing과 pause_prompt에는 질문을 덧붙이지 않는다.

    Args:
        state:
            현재 질문, 턴 상황, 기존 transcript와 선택적인 last_signal을 가진
            세션 상태. last_signal은 LLM preamble 생성 맥락으로 전달한다.

        runtime:
            선택적 LLM이 담긴 InterviewDeps를 제공하는 LangGraph runtime.

    Returns:
        조립된 last_utterance, 안내 문장과 질문 원문이 분리된
        utterance_queue, 면접관 Turn이 추가된 transcript를 담은 부분 상태.
        질문이 필요한 상황인데 현재 질문이 없으면 error를 반환한다.
    """
    turn_type = _state_get(state, "turn_type", "question")
    current_question = _state_get(state, "current_question")
    if isinstance(current_question, dict):
        current_question = Question.model_validate(current_question)

    includes_question = turn_type not in {"closing", "pause_prompt"}
    if includes_question and current_question is None:
        return {"error": "면접관 발화를 만들 현재 질문이 없습니다."}

    question_kind = current_question.kind if current_question is not None else None
    reaction_policy = select_reaction_policy(_state_get(state, "last_signal"))
    fallback_preamble = _select_utterance_preamble(
        turn_type,
        question_kind,
        reaction_policy,
    )
    preamble = fallback_preamble
    preamble_source = "template"
    deps = _runtime_deps(runtime)
    if deps.llm is not None:
        try:
            preamble = _generate_llm_preamble(
                llm=deps.llm,
                state=state,
                turn_type=turn_type,
                current_question=current_question,
                reaction_policy=reaction_policy,
            )
            preamble_source = "llm"
        except Exception as exc:
            preamble = fallback_preamble
            preamble_source = "template_fallback"
            log_llm_error(
                "INTERVIEWER_PREAMBLE",
                exc,
                metadata={
                    "turn_type": turn_type,
                    "question_id": current_question.question_id if current_question else None,
                    "question_kind": question_kind.value if question_kind else None,
                    "reaction_tone": reaction_policy.tone.value,
                    "answer_quality": (
                        reaction_policy.quality.value if reaction_policy.quality else None
                    ),
                },
                fallback={"preamble": fallback_preamble},
            )

    last_utterance = preamble
    question_text = None
    if includes_question and current_question is not None:
        question_text = current_question.text
        last_utterance = f"{preamble}\n\n{question_text}"

    interviewer_turn = Turn(
        role="interviewer",
        text=last_utterance,
        question_id=current_question.question_id if includes_question else None,
        kind=current_question.kind.value if includes_question else None,
    )
    transcript = _state_get(state, "transcript", []) or []

    log_llm_output(
        "INTERVIEWER_UTTERANCE",
        {
            "preamble": preamble,
            "question": question_text,
            "utterance": last_utterance,
        },
        metadata={
            "source": preamble_source,
            "turn_type": turn_type,
            "question_id": current_question.question_id if current_question else None,
            "question_kind": question_kind.value if question_kind else None,
            "reaction_tone": reaction_policy.tone.value,
            "answer_quality": (
                reaction_policy.quality.value if reaction_policy.quality else None
            ),
        },
        status=preamble_source,
    )

    return {
        "last_utterance": last_utterance,
        "utterance_queue": _build_utterance_queue(preamble, question_text),
        "transcript": [*transcript, interviewer_turn],
        "error": None,
    }


def after_compose_utterance(state: SessionState | dict[str, Any]) -> str:
    """발화 조립 후 사용자 입력을 기다릴지 세션을 끝낼지 결정한다.

    일반 질문과 재제시 발화 뒤에는 wait_event로 이동한다. finalize를 거쳐
    finished=True가 된 종료 발화 뒤에는 더 이상 사용자 입력을 기다리지 않고
    그래프를 종료한다. 이 함수는 상태를 읽기만 하고 변경하지 않는다.

    Args:
        state:
            finished 종료 여부를 가진 현재 세션 상태.

    Returns:
        계속 입력을 기다리면 "wait_event", 종료하면 "end".
    """
    return "end" if _state_get(state, "finished", False) else "wait_event"
