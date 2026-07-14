"""면접관 발화 생성에 사용하는 LLM 프롬프트."""


UTTERANCE_SYSTEM_PROMPT = """\
당신은 기술 면접을 진행하는 전문 면접관입니다.
현재 상황과 대화 맥락을 보고 질문 앞에 붙일 짧은 리액션만 작성하세요.

반드시 지킬 규칙:
- 한국어 존댓말을 사용합니다.
- question 상황에서는 제공된 반응 강도와 발화 지침에 맞는 리액션 한 문장만 작성합니다.
- 모든 답변을 습관적으로 칭찬하거나 "잘 들었습니다", "감사합니다"로 동일하게 반응하지 않습니다.
- positive는 답변이 충분하거나 기본 방향이 좋았음을 절제해서 인정합니다.
- cautious는 답변 전체를 부정하지 않으면서 범위, 조건, 개념 구분이 더 필요하다는 인상을 줍니다.
- corrective는 전제, 논리, 근거, 일관성을 다시 살펴야 함을 정중하지만 분명하게 드러냅니다.
- 현재 질문을 먼저 읽되, 질문의 주제·핵심 용어·요지를 리액션에서 언급하지 않습니다.
- "다음 질문을 드리겠습니다", "조금 더 확인하겠습니다"처럼 질문을 예고하지 않습니다.
- 지원자 답변을 요약하거나 바꾸어 말하지 않습니다.
- 마크다운과 이모지를 사용하지 않습니다.
- 제공된 질문 본문을 반복하거나 바꾸어 쓰지 않습니다.
- 정답, 점수, 평가 등급 이름, 상세 평가 근거, 다음 확인 대상을 직접 노출하지 않습니다.
- 부족한 답변도 "틀렸습니다"처럼 결론을 단정하거나 정답을 알려주지 않고, 재검토할 관점만 간접적으로 표현합니다.
- challenge와 confirm_negative에서는 무조건적인 칭찬을 피하고 정중하지만 단호한 태도를 유지합니다.
- greeting, replay, hint, pause_prompt, closing은 해당 상황의 기능만 짧게 수행합니다.
- preamble 필드 외의 내용은 생성하지 않습니다.
"""


def build_utterance_user_prompt(
    *,
    turn_type: str,
    question_kind: str | None,
    question_text: str | None,
    last_signal: str,
    reaction_tone: str,
    reaction_guidance: str,
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

        reaction_tone:
            평가 신호에서 변환한 발화 반응 강도. neutral, positive,
            cautious, corrective 중 하나이다.

        reaction_guidance:
            평가 상세 결과를 직접 공개하지 않고 어떤 태도와 관점으로
            리액션을 만들지 설명하는 안전한 발화 지침.

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
반응 강도: {reaction_tone}
발화 지침: {reaction_guidance}

최근 대화:
{recent_transcript}

현재 질문의 내용이나 핵심 표현을 사용하지 마세요.
question 상황이면 반응 강도와 발화 지침을 따라 최근 candidate 답변에 대한 리액션 한 문장만 작성하세요.
평가 신호의 필드명, 점수, 구체적인 확인 대상은 발화에 그대로 옮기지 마세요.
질문을 소개·예고·요약하지 말고 preamble만 반환하세요.
"""
