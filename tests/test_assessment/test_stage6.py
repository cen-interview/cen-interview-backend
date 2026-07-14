from interview.assessment.scoring import AnswerAttempt, score_question_set
from interview.schemas.question import Difficulty, QuestionCategory, QuestionKind
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


def make_attempt(
    *,
    question_id: str,
    question_text: str,
    answer_text: str,
    kind: QuestionKind,
    quality: AnswerQuality,
    accuracy: float,
    sufficiency: float,
) -> AnswerAttempt:
    signal = AnswerQualitySignal(
        answer_id=f"answer-{question_id}",
        question_id=question_id,
        quality=quality,
        rationale=[
            f"{kind.value} 답변을 {quality.value}로 판정했습니다."
        ],
        accuracy=accuracy,
        sufficiency=sufficiency,
    )

    return AnswerAttempt(
        answer_id=signal.answer_id,
        question_id=question_id,
        question_text=question_text,
        question_topic="JPA",
        question_kind=kind,
        question_category=QuestionCategory.TECHNICAL,
        question_difficulty=Difficulty.MEDIUM,
        answer_text=answer_text,
        signal=signal,
    )


def print_scenario(
    label: str,
    attempts: list[AnswerAttempt],
    score: float,
    comment: str,
) -> None:
    print(f"\n===== {label} =====")

    for index, attempt in enumerate(attempts, start=1):
        print(f"\n[{index}. {attempt.question_kind.value.upper()}]")
        print("질문:", attempt.question_text)
        print("답변:", attempt.answer_text)
        print("quality:", attempt.signal.quality)
        print("accuracy:", attempt.signal.accuracy)
        print("sufficiency:", attempt.signal.sufficiency)

    print("\n최종 점수:", score)
    print("평가 comment:", comment)


def test_stage6_scenario_2_scores_clearly_higher_than_scenario_4() -> None:
    scenario_2_attempts = [
        make_attempt(
            question_id="scenario-2-main",
            question_text="JPA N+1 문제의 원인과 해결 방법을 설명해 주세요.",
            answer_text=(
                "연관 엔티티에 접근할 때 엔티티마다 "
                "추가 쿼리가 발생하는 문제입니다."
            ),
            kind=QuestionKind.MAIN,
            quality=AnswerQuality.BONUS_AVAILABLE,
            accuracy=0.8,
            sufficiency=0.5,
        ),
        make_attempt(
            question_id="scenario-2-follow-up",
            question_text="N+1 문제를 어떤 방법으로 해결할 수 있나요?",
            answer_text=(
                "fetch join이나 EntityGraph로 한 번에 조회하고, "
                "상황에 따라 batch size를 설정할 수 있습니다."
            ),
            kind=QuestionKind.FOLLOW_UP,
            quality=AnswerQuality.SUFFICIENT,
            accuracy=0.9,
            sufficiency=0.9,
        ),
    ]

    scenario_4_attempts = [
        make_attempt(
            question_id="scenario-4-main",
            question_text="JPA 지연 로딩의 장단점과 주의할 점을 설명해 주세요.",
            answer_text=(
                "지연 로딩은 쿼리 수를 줄이기 때문에 "
                "즉시 로딩보다 항상 유리합니다."
            ),
            kind=QuestionKind.MAIN,
            quality=AnswerQuality.MISCONCEPTION,
            accuracy=0.4,
            sufficiency=0.7,
        ),
        make_attempt(
            question_id="scenario-4-challenge",
            question_text=(
                "N+1 문제가 발생할 수 있는데도 "
                "지연 로딩이 항상 유리하다고 볼 수 있나요?"
            ),
            answer_text=(
                "네. 지연 로딩은 어떤 상황에서도 "
                "쿼리 수를 줄이므로 항상 유리합니다."
            ),
            kind=QuestionKind.CHALLENGE,
            quality=AnswerQuality.MISCONCEPTION,
            accuracy=0.4,
            sufficiency=0.7,
        ),
    ]

    scenario_2 = score_question_set(
        scenario_2_attempts
    )
    scenario_4 = score_question_set(
        scenario_4_attempts
    )

    print_scenario(
        "시나리오 2: 꼬리질문에서 보완",
        scenario_2_attempts,
        scenario_2.score,
        scenario_2.comment,
    )
    print_scenario(
        "시나리오 4: 오개념 미정정",
        scenario_4_attempts,
        scenario_4.score,
        scenario_4.comment,
    )

    score_difference = (
        scenario_2.score
        - scenario_4.score
    )

    print("\n===== 완료 기준 =====")
    print("시나리오 2 점수:", scenario_2.score)
    print("시나리오 4 점수:", scenario_4.score)
    print("점수 차이:", score_difference)

    assert scenario_2.score > scenario_4.score
    assert score_difference >= 20.0
