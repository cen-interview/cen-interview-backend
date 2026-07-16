"""Generate shareable technical-answer rubric candidates."""

from interview.llm.client import get_llm
from interview.llm.logging import log_llm_error, log_llm_output
from interview.schemas.question import Question
from interview.schemas.rubric import RubricCandidate


RUBRIC_EXTRACTION_SYSTEM = """\
당신은 기술 면접 답변에서 재사용 가능한 정답 평가 기준을 추출하는 도우미다.

질문과 충분하다고 평가된 답변을 보고, 다른 답변을 평가할 때 사용할 핵심 정답 요소를
3~5개 추출하라.

규칙:
- 특정 사용자의 이름, 프로젝트명, 개인정보는 포함하지 않는다.
- 답변 원문을 그대로 저장하지 않는다.
- 기술적으로 검증 가능한 핵심 개념과 필수 설명만 criteria에 작성한다.
- 각 criterion은 독립적으로 답변에 포함되었는지 판단할 수 있어야 한다.
- 질문에 직접 답하는 핵심 개념 2~3개만 required=true로 설정한다.
- 예시, 장점, 활용 환경 같은 보충 내용은 required=false로 설정한다.
- 질문에 대한 정답 요소만 작성하고 평가 코멘트는 작성하지 않는다.
"""


def generate_rubric_candidate(
    question: Question,
    answer_context: str,
) -> RubricCandidate | None:
    """Create a rubric candidate from a sufficient technical question set."""
    prompt = f"""
[질문]
{question.text}

[주제]
{question.topic}

[질문 세트 답변]
{answer_context}

위 질문에 답하기 위해 필요한 핵심 정답 요소를 추출하라.
"""

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(RubricCandidate)
        result = structured_llm.invoke(
            [
                {"role": "system", "content": RUBRIC_EXTRACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
        result = result.model_copy(
            update={
                "question_id": question.question_id,
                "topic": question.topic,
                "question": question.text,
            }
        )
        log_llm_output(
            "TECHNICAL_RUBRIC_EXTRACTION",
            result,
            metadata={
                "question_id": question.question_id,
                "topic": question.topic,
                "criterion_count": len(result.criteria),
            },
        )
        return result if result.criteria else None
    except Exception as exc:
        log_llm_error(
            "TECHNICAL_RUBRIC_EXTRACTION",
            exc,
            metadata={
                "question_id": question.question_id,
                "topic": question.topic,
            },
            input_data={"question": question.text},
        )
        return None
