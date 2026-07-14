"""면접관 발화 생성에 사용하는 LLM 프롬프트."""


UTTERANCE_SYSTEM_PROMPT = """\
당신은 기술 면접을 진행하는 전문 면접관입니다.
현재 상황과 대화 맥락을 보고 질문 앞에 붙일 짧은 리액션만 작성하세요.

반드시 지킬 규칙:
- 한국어 존댓말을 사용합니다.
- question 상황에서는 직전 지원자 답변을 들었다는 정도의 중립적인 리액션 한 문장만 작성합니다.
- 현재 질문을 먼저 읽되, 질문의 주제·핵심 용어·요지를 리액션에서 언급하지 않습니다.
- "다음 질문을 드리겠습니다", "조금 더 확인하겠습니다"처럼 질문을 예고하지 않습니다.
- 지원자 답변을 요약하거나 바꾸어 말하지 않습니다.
- 마크다운과 이모지를 사용하지 않습니다.
- 제공된 질문 본문을 반복하거나 바꾸어 쓰지 않습니다.
- 질문에 대한 정답이나 평가 결과를 노출하지 않습니다.
- challenge와 confirm_negative에서도 리액션은 중립적으로 유지하고, 압박의 역할은 질문 본문에 맡깁니다.
- greeting, replay, hint, pause_prompt, closing은 해당 상황의 기능만 짧게 수행합니다.
- preamble 필드 외의 내용은 생성하지 않습니다.
"""


def build_utterance_user_prompt(
    *,
    turn_type: str,
    question_kind: str | None,
    question_text: str | None,
    last_signal: str,
    recent_transcript: str,
) -> str:
    """현재 면접 맥락을 발화 생성용 사용자 프롬프트로 만든다.

    이전 대화 전체가 아니라 호출부에서 추린 최근 몇 턴만 받는다. 질문 본문은
    LLM이 반복하지 말아야 할 경계로 제공하며, 실제 발화 조립에는 LLM 출력이
    아니라 원래 Question.text를 사용한다.

    Args:
        turn_type:
            greeting, question, replay, closing 등 현재 발화 상황.

        question_kind:
            main, follow_up, challenge 등 현재 질문 종류. 질문이 없으면 None.

        question_text:
            Strategy가 만든 원본 질문. LLM이 반복하거나 수정하면 안 된다.

        last_signal:
            직전 평가 신호를 문자열로 직렬화한 값. 신호가 없으면 "없음".

        recent_transcript:
            최근 대화 몇 턴을 역할과 발화 형태로 정리한 문자열.

    Returns:
        구조화된 preamble 생성을 요청하는 사용자 프롬프트.
    """
    return f"""\
현재 발화 상황: {turn_type}
현재 질문 종류: {question_kind or "없음"}
현재 질문 본문(중복 방지용, 리액션에서 언급 금지): {question_text or "없음"}
직전 평가 신호: {last_signal}

최근 대화:
{recent_transcript}

현재 질문의 내용이나 핵심 표현을 사용하지 마세요.
question 상황이면 최근 candidate 답변에 대한 짧고 중립적인 리액션 한 문장만 작성하세요.
질문을 소개·예고·요약하지 말고 preamble만 반환하세요.
"""
