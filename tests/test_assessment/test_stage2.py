"""실제 LLM judge 테스트.

좋은 답/얕은 답/틀린 답 3종이 상식적인 quality로 갈리는지 확인한다.
이 테스트는 실행할 때마다 실제 LLM을 호출한다.
"""

from interview.assessment import evaluator
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)
from interview.schemas.signals import AnswerQuality


def make_question() -> Question:
    return Question(
        question_id="q-real-llm-jpa-n-plus-one",
        text="JWT 기반 인증의 동작 방식과 주의할 점을 설명해 주세요.",
        topic="인증",
        difficulty=Difficulty.MEDIUM,
        kind=QuestionKind.MAIN,
        category=QuestionCategory.TECHNICAL,
    )


def test_real_llm_judge_good_shallow_wrong_answers_diverge() -> None:
    question = make_question()

    good = evaluator.judge_answer(
        question=question,
        answer_text=(
            "JWT는 로그인 후 서버가 사용자 정보를 담은 토큰을 서명해서 발급하고, "
            "클라이언트가 이후 요청마다 Authorization 헤더에 담아 보내는 방식입니다. "
            "서버는 토큰의 서명을 검증해 사용자를 식별합니다. 주의할 점은 토큰은 탈취되면 만료 전까지 악용될 수 있으므로 "
            "만료 시간을 짧게 두고, "
            "refresh token을 분리해 관리하며, 민감 정보를 payload에 넣지 않는 것입니다."
        ),
    )

    shallow = evaluator.judge_answer( 
        question=question,
        answer_text="JWT는 로그인할 때 쓰는 토큰입니다.",
    )

    wrong = evaluator.judge_answer(
        question=question,
        answer_text="JWT는 서버 세션에 사용자 정보를 저장하고, 클라이언트는 세션 ID만 들고 있는 방식입니다.",
    )

    def print_signal(label: str, signal) -> None: 
        print(f"\n===== {label} =====")
        print("quality:", signal.quality)
        print("accuracy:", signal.accuracy)
        print("sufficiency:", signal.sufficiency)
        print("next_probe_target:", signal.next_probe_target)
        print("rationale:", signal.rationale)


    print_signal("GOOD ANSWER", good)
    print_signal("SHALLOW ANSWER", shallow)
    print_signal("WRONG ANSWER", wrong)

