import pytest

from interview.assessment.scoring import (
    AnswerAttempt,
    _apply_difficulty_adjustment,
    _apply_history_adjustment,
    _calculate_attempt_base_score,
    _calculate_set_delivery_score,
    _combine_content_and_delivery_score,
    score_question_set,
)
from interview.assessment.scoring_policy import (
    EXPECTED_PRIOR_QUALITY_BY_KIND,
    RESOLVED_RATE_BY_KIND,
    UNRESOLVED_RATE_BY_KIND,
)
from interview.schemas.question import Difficulty, QuestionCategory, QuestionKind
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


def make_attempt(
    *,
    question_id: str,
    kind: QuestionKind,
    quality: AnswerQuality,
    accuracy: float = 0.8,
    sufficiency: float = 0.8,
    difficulty: Difficulty = Difficulty.MEDIUM,
    delivery_metrics: dict | None = None,
) -> AnswerAttempt:
    signal = AnswerQualitySignal(
        answer_id=f"answer-{question_id}",
        question_id=question_id,
        quality=quality,
        rationale=[f"{kind.value} / {quality.value} 테스트 판정"],
        accuracy=accuracy,
        sufficiency=sufficiency,
    )

    return AnswerAttempt(
        answer_id=signal.answer_id,
        question_id=question_id,
        question_text="테스트 질문입니다.",
        question_topic="테스트 주제",
        question_kind=kind,
        question_category=QuestionCategory.TECHNICAL,
        question_difficulty=difficulty,
        answer_text="테스트 답변입니다.",
        signal=signal,
        delivery_metrics=delivery_metrics,
    )


def print_scoring_process(
    label: str,
    attempts: list[AnswerAttempt],
) -> None:
    last_attempt = attempts[-1]
    accuracy = last_attempt.signal.accuracy
    sufficiency = last_attempt.signal.sufficiency
    effective_sufficiency = accuracy * sufficiency

    base_score = _calculate_attempt_base_score(
        last_attempt
    )
    history_score, history_comment = (
        _apply_history_adjustment(
            base_score=base_score,
            attempts=attempts,
        )
    )
    difficulty_score, difficulty_multiplier = (
        _apply_difficulty_adjustment(
            score=history_score,
            attempts=attempts,
        )
    )
    delivery_score = _calculate_set_delivery_score(
        attempts
    )
    final_score = _combine_content_and_delivery_score(
        content_score=difficulty_score,
        delivery_score=delivery_score,
    )

    history_rate = 0.0
    if len(attempts) >= 2:
        last_kind = last_attempt.question_kind
        resolved = (
            last_attempt.signal.quality
            == AnswerQuality.SUFFICIENT
        )
        expected_quality = (
            EXPECTED_PRIOR_QUALITY_BY_KIND.get(
                last_kind
            )
        )
        has_expected_history = (
            last_kind == QuestionKind.HINT
            or (
                expected_quality is not None
                and any(
                    attempt.signal.quality
                    == expected_quality
                    for attempt in attempts[:-1]
                )
            )
        )

        if has_expected_history:
            history_rate = (
                RESOLVED_RATE_BY_KIND[last_kind]
                if resolved
                else UNRESOLVED_RATE_BY_KIND[last_kind]
            )

    print(f"\n===== {label} =====")
    print("[1. ATTEMPT FLOW]")
    for attempt in attempts:
        print(
            f"{attempt.question_kind.value}"
            f"({attempt.signal.quality.value})"
            f" accuracy={attempt.signal.accuracy}"
            f" sufficiency={attempt.signal.sufficiency}"
        )

    print("\n[2. BASE SCORE]")
    print("last_accuracy:", accuracy)
    print("last_sufficiency:", sufficiency)
    print("effective_sufficiency:", round(effective_sufficiency, 4))
    print(
        "formula:",
        f"{accuracy} * (0.7 + 0.3 * {sufficiency}) * 100",
    )
    print("base_score:", base_score)

    print("\n[3. HISTORY ADJUSTMENT]")
    print("history_rate:", f"{history_rate * 100:+.2f}%")
    print("history_score:", history_score)
    print("comment:", history_comment)

    print("\n[4. DIFFICULTY ADJUSTMENT]")
    print("difficulty:", attempts[0].question_difficulty.value)
    print("multiplier:", difficulty_multiplier)
    print("content_score:", difficulty_score)

    print("\n[5. DELIVERY ADJUSTMENT]")
    print("delivery_score:", delivery_score)
    print(
        "formula:",
        "content 그대로"
        if delivery_score is None
        else "content * 0.9 + delivery * 0.1",
    )

    print("\n[6. FINAL SCORE]")
    print("final_score:", final_score)


@pytest.mark.parametrize(
    (
        "prior_quality",
        "derived_kind",
        "unresolved_quality",
    ),
    [
        (
            AnswerQuality.BONUS_AVAILABLE,
            QuestionKind.FOLLOW_UP,
            AnswerQuality.BONUS_AVAILABLE,
        ),
        (
            AnswerQuality.MISCONCEPTION,
            QuestionKind.CHALLENGE,
            AnswerQuality.MISCONCEPTION,
        ),
        (
            AnswerQuality.CONFIRM_POSITIVE,
            QuestionKind.CONFIRM_POSITIVE,
            AnswerQuality.CONFIRM_POSITIVE,
        ),
        (
            AnswerQuality.CONFIRM_NEGATIVE,
            QuestionKind.CONFIRM_NEGATIVE,
            AnswerQuality.CONFIRM_NEGATIVE,
        ),
        (
            AnswerQuality.TRAP_AVAILABLE,
            QuestionKind.TRAP,
            AnswerQuality.TRAP_AVAILABLE,
        ),
    ],
)
def test_stage6_5_resolved_set_scores_higher_than_unresolved_set(
    prior_quality: AnswerQuality,
    derived_kind: QuestionKind,
    unresolved_quality: AnswerQuality,
) -> None:
    main_attempt = make_attempt(
        question_id=f"main-{derived_kind.value}",
        kind=QuestionKind.MAIN,
        quality=prior_quality,
    )
    resolved_attempt = make_attempt(
        question_id=f"resolved-{derived_kind.value}",
        kind=derived_kind,
        quality=AnswerQuality.SUFFICIENT,
    )
    unresolved_attempt = make_attempt(
        question_id=f"unresolved-{derived_kind.value}",
        kind=derived_kind,
        quality=unresolved_quality,
    )

    resolved = score_question_set(
        [main_attempt, resolved_attempt]
    )
    unresolved = score_question_set(
        [main_attempt, unresolved_attempt]
    )

    print_scoring_process(
        f"{derived_kind.value.upper()} RESOLVED",
        [main_attempt, resolved_attempt],
    )
    print_scoring_process(
        f"{derived_kind.value.upper()} UNRESOLVED",
        [main_attempt, unresolved_attempt],
    )

    assert resolved.score > unresolved.score


def test_stage6_5_hint_recovery_scores_higher_than_hint_failure() -> None:
    main_attempt = make_attempt(
        question_id="main-hint",
        kind=QuestionKind.MAIN,
        quality=AnswerQuality.BONUS_AVAILABLE,
    )
    recovered_attempt = make_attempt(
        question_id="hint-recovered",
        kind=QuestionKind.HINT,
        quality=AnswerQuality.SUFFICIENT,
    )
    failed_attempt = make_attempt(
        question_id="hint-failed",
        kind=QuestionKind.HINT,
        quality=AnswerQuality.MISCONCEPTION,
    )

    recovered = score_question_set(
        [main_attempt, recovered_attempt]
    )
    failed = score_question_set(
        [main_attempt, failed_attempt]
    )

    print_scoring_process(
        "HINT RECOVERED",
        [main_attempt, recovered_attempt],
    )
    print_scoring_process(
        "HINT FAILED",
        [main_attempt, failed_attempt],
    )

    assert recovered.score > failed.score


def test_stage6_5_scenario_2_scores_clearly_higher_than_scenario_4() -> None:
    scenario_2 = score_question_set(
        [
            make_attempt(
                question_id="scenario-2-main",
                kind=QuestionKind.MAIN,
                quality=AnswerQuality.BONUS_AVAILABLE,
                accuracy=0.8,
                sufficiency=0.5,
            ),
            make_attempt(
                question_id="scenario-2-follow-up",
                kind=QuestionKind.FOLLOW_UP,
                quality=AnswerQuality.SUFFICIENT,
                accuracy=0.9,
                sufficiency=0.9,
            ),
        ]
    )
    scenario_4 = score_question_set(
        [
            make_attempt(
                question_id="scenario-4-main",
                kind=QuestionKind.MAIN,
                quality=AnswerQuality.MISCONCEPTION,
                accuracy=0.4,
                sufficiency=0.7,
            ),
            make_attempt(
                question_id="scenario-4-challenge",
                kind=QuestionKind.CHALLENGE,
                quality=AnswerQuality.MISCONCEPTION,
                accuracy=0.4,
                sufficiency=0.7,
            ),
        ]
    )

    print_scoring_process(
        "SCENARIO 2: BONUS THEN RESOLVED",
        [
            make_attempt(
                question_id="scenario-2-print-main",
                kind=QuestionKind.MAIN,
                quality=AnswerQuality.BONUS_AVAILABLE,
                accuracy=0.8,
                sufficiency=0.5,
            ),
            make_attempt(
                question_id="scenario-2-print-follow-up",
                kind=QuestionKind.FOLLOW_UP,
                quality=AnswerQuality.SUFFICIENT,
                accuracy=0.9,
                sufficiency=0.9,
            ),
        ],
    )
    print_scoring_process(
        "SCENARIO 4: MISCONCEPTION UNRESOLVED",
        [
            make_attempt(
                question_id="scenario-4-print-main",
                kind=QuestionKind.MAIN,
                quality=AnswerQuality.MISCONCEPTION,
                accuracy=0.4,
                sufficiency=0.7,
            ),
            make_attempt(
                question_id="scenario-4-print-challenge",
                kind=QuestionKind.CHALLENGE,
                quality=AnswerQuality.MISCONCEPTION,
                accuracy=0.4,
                sufficiency=0.7,
            ),
        ],
    )

    print("\n===== SCENARIO SCORE ORDERING =====")
    print("scenario_2_score:", scenario_2.score)
    print("scenario_4_score:", scenario_4.score)
    print("score_difference:", scenario_2.score - scenario_4.score)

    assert scenario_2.score == pytest.approx(91.67)
    assert scenario_4.score == pytest.approx(25.48)
    assert scenario_2.score - scenario_4.score >= 20.0


def test_stage6_1_sufficiency_cannot_score_without_accuracy() -> None:
    result = score_question_set(
        [
            make_attempt(
                question_id="zero-accuracy",
                kind=QuestionKind.MAIN,
                quality=AnswerQuality.MISCONCEPTION,
                accuracy=0.0,
                sufficiency=1.0,
            )
        ]
    )

    assert result.score == 0.0


def test_stage6_difficulty_order_is_hard_medium_easy() -> None:
    def score_for(difficulty: Difficulty) -> float:
        return score_question_set(
            [
                make_attempt(
                    question_id=f"difficulty-{difficulty.value}",
                    kind=QuestionKind.MAIN,
                    quality=AnswerQuality.SUFFICIENT,
                    difficulty=difficulty,
                )
            ]
        ).score

    easy_score = score_for(Difficulty.EASY)
    medium_score = score_for(Difficulty.MEDIUM)
    hard_score = score_for(Difficulty.HARD)

    print("\n===== DIFFICULTY =====")
    print("easy:", easy_score)
    print("medium:", medium_score)
    print("hard:", hard_score)

    assert hard_score > medium_score > easy_score


def test_stage6_delivery_is_applied_only_when_metrics_exist() -> None:
    chat_attempt = make_attempt(
        question_id="delivery-chat",
        kind=QuestionKind.MAIN,
        quality=AnswerQuality.SUFFICIENT,
        delivery_metrics=None,
    )
    voice_attempt = make_attempt(
        question_id="delivery-voice",
        kind=QuestionKind.MAIN,
        quality=AnswerQuality.SUFFICIENT,
        delivery_metrics={
            "speech_rate_wpm": 185,
            "filler_count": 7,
        },
    )

    chat = score_question_set([chat_attempt])
    voice = score_question_set([voice_attempt])

    print("\n===== DELIVERY =====")
    print("chat_score:", chat.score)
    print("voice_score:", voice.score)
    print_scoring_process(
        "CHAT SCORE",
        [chat_attempt],
    )
    print_scoring_process(
        "VOICE SCORE",
        [voice_attempt],
    )

    assert chat.score == pytest.approx(75.2)
    assert voice.score == pytest.approx(75.28)


def test_stage6_empty_attempts_returns_zero() -> None:
    result = score_question_set([])

    assert result.score == 0.0
