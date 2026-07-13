"""음성 답변에 포함된 짧은 면접 제어 명령을 해석한다.

STT가 모든 발화를 답변 제출로 전달하더라도, 사용자가 짧게 말한 종료 또는
다시 듣기 요청은 Interviewer 이벤트로 바꿀 수 있어야 한다. 반면 기술 답변에
우연히 포함된 "면접 종료 조건" 같은 표현은 명령으로 오인하지 않아야 한다.

이 모듈은 문자열 정규화와 명령 의도 판정만 담당한다. 실제 이벤트 변환은
세션 모드를 알고 있는 ``validate_event`` 노드에서 수행한다.
"""

import re
import unicodedata
from typing import Literal

VoiceCommand = Literal["end", "replay"]
"""음성 발화에서 식별할 수 있는 면접 제어 명령 종류."""

MAX_SHORT_UTTERANCE_LENGTH = 30
"""공백을 제외한 발화를 짧은 명령 후보로 보는 최대 문자 수."""

# 짧은 발화 전체가 아래 표현 중 하나일 때만 명령으로 판정한다. 단순히
# "종료" 또는 "다시"가 포함됐다는 이유만으로 매칭하지 않도록 fullmatch한다.
END_PATTERNS: tuple[str, ...] = (
    r"(?:네\s*|예\s*)?(?:이제\s*)?(?:면접(?:을)?\s*)?(?:종료|그만|끝)",
    r"(?:네\s*|예\s*)?(?:이제\s*)?(?:면접(?:을)?\s*)?"
    r"(?:종료|그만)(?:\s*(?:해\s*주세요|해주세요|할게요|하겠습니다|할래요|하죠))",
    r"(?:네\s*|예\s*)?(?:이제\s*)?(?:면접(?:을)?\s*)?"
    r"끝(?:내\s*주세요|낼게요|내겠습니다)",
    r"(?:네\s*|예\s*)?(?:이제\s*)?(?:면접(?:을)?\s*)?"
    r"마치(?:겠습니다|고\s*싶습니다|도록\s*할게요|게\s*해주세요)",
    r"(?:네\s*|예\s*)?(?:여기까지|이만)(?:\s*(?:할게요|하겠습니다|해주세요))?",
)

REPLAY_PATTERNS: tuple[str, ...] = (
    r"(?:네\s*|예\s*)?(?:질문(?:을)?\s*)?다시",
    r"(?:네\s*|예\s*)?(?:질문(?:을)?\s*)?(?:다시|한\s*번\s*더)\s*"
    r"(?:말씀|말|설명)\s*(?:해\s*주세요|해주세요|해\s*주시겠어요)",
    r"(?:네\s*|예\s*)?(?:질문(?:을)?\s*)?(?:다시|한\s*번\s*더)\s*"
    r"(?:들려\s*주세요|읽어\s*주세요)",
    r"(?:잘\s*)?못\s*들었(?:어요|습니다)",
)

# 긴 발화에는 명령형이 분명한 패턴만 허용한다. 예를 들어 명사구인
# "면접 종료"는 제외하므로 "면접 종료 조건을 구현했습니다"가 시작 부분에
# 있어도 종료 명령이 되지 않는다.
_EXPLICIT_END_PATTERNS: tuple[str, ...] = END_PATTERNS[1:]
_EXPLICIT_REPLAY_PATTERNS: tuple[str, ...] = REPLAY_PATTERNS[1:]


def normalize_utterance(text: str) -> str:
    """STT 발화를 명령 패턴 비교에 적합한 문자열로 정규화한다.

    유니코드 호환 문자를 통일하고 영문 대소문자 차이를 없앤 뒤, 문장 부호를
    공백으로 바꾼다. 마지막으로 연속 공백을 하나로 합쳐 STT가 쉼표나 마침표를
    붙였는지와 관계없이 같은 명령을 같은 문자열로 비교할 수 있게 한다.

    Args:
        text:
            STT가 만든 원본 발화 문자열.

    Returns:
        유니코드, 대소문자, 문장 부호와 연속 공백을 정리한 발화 문자열.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w\s가-힣]", " ", normalized)
    return " ".join(normalized.split())


def detect_voice_command(text: str) -> VoiceCommand | None:
    """음성 발화가 종료 또는 다시 듣기 명령인지 판정한다.

    짧은 발화는 명령 패턴과 발화 전체가 일치할 때 판정한다. 최대 길이를 넘는
    발화는 기술 답변일 가능성이 높으므로, 명령형이 분명한 패턴이 발화의 시작
    또는 끝 경계에 있을 때만 판정한다. 어느 쪽도 확실하지 않으면 일반 답변을
    유지할 수 있도록 None을 반환한다.

    Args:
        text:
            판정할 STT 답변 텍스트.

    Returns:
        종료 명령이면 ``"end"``, 다시 듣기 명령이면 ``"replay"``.
        명령 의도가 확실하지 않으면 None.
    """
    normalized = normalize_utterance(text)
    if not normalized:
        return None

    compact_length = len(normalized.replace(" ", ""))
    if compact_length <= MAX_SHORT_UTTERANCE_LENGTH:
        if _fullmatches_any(normalized, END_PATTERNS):
            return "end"
        if _fullmatches_any(normalized, REPLAY_PATTERNS):
            return "replay"

    if _matches_boundary(normalized, _EXPLICIT_END_PATTERNS):
        return "end"
    if _matches_boundary(normalized, _EXPLICIT_REPLAY_PATTERNS):
        return "replay"
    return None


def _fullmatches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """문자열 전체와 일치하는 명령 패턴이 있는지 확인한다.

    Args:
        text:
            정규화된 발화 문자열.

        patterns:
            비교할 정규 표현식 패턴 목록.

    Returns:
        하나 이상의 패턴이 문자열 전체와 일치하면 True.
    """
    return any(re.fullmatch(pattern, text) is not None for pattern in patterns)


def _matches_boundary(text: str, patterns: tuple[str, ...]) -> bool:
    """명령 패턴이 발화의 시작 또는 끝 경계에 있는지 확인한다.

    시작 패턴 뒤와 끝 패턴 앞에 공백 또는 문자열 경계가 있어야 한다. 따라서
    명령과 철자가 일부만 같은 긴 단어가 잘못 매칭되는 것을 방지한다.

    Args:
        text:
            정규화된 발화 문자열.

        patterns:
            경계에서 비교할 명시적 명령 패턴 목록.

    Returns:
        하나 이상의 패턴이 발화 시작 또는 끝에 있으면 True.
    """
    for pattern in patterns:
        if re.match(rf"(?:{pattern})(?=$|\s)", text) is not None:
            return True
        if re.search(rf"(?<!\S)(?:{pattern})$", text) is not None:
            return True
    return False
