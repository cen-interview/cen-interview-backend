from interview.assessment.agent import AssessmentAgent
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)


def make_question(question_id: str) -> Question:
    return Question(
        question_id=question_id,
        text="JPA N+1 문제의 원인과 해결 방법을 설명해 주세요.",
        topic="JPA N+1",
        difficulty=Difficulty.MEDIUM,
        kind=QuestionKind.MAIN,
        category=QuestionCategory.TECHNICAL,
    )


def evaluate_as_interviewer(
    *,
    question_id: str,
    answer_text: str,
    delivery_metrics: dict | None,
):
    """Interviewer처럼 답변과 선택적 음성 지표를 Assessment에 전달한다."""

    assessment = AssessmentAgent()
    question = make_question(question_id)

    signal = assessment.evaluate(
        question=question,
        answer_text=answer_text,
        delivery_metrics=delivery_metrics,
    )
    assessment.complete_question_set(
        main_question_id=question.question_id,
    )

    return signal, assessment.evaluations[0]


def print_result(label: str, signal, evaluation) -> None:
    print(f"\n===== {label} =====")
    print("quality:", signal.quality)
    print("accuracy:", signal.accuracy)
    print("sufficiency:", signal.sufficiency)
    print("next_probe_target:", signal.next_probe_target)
    print("rationale:", signal.rationale)
    print("signal.delivery_note:", signal.delivery_note)
    print("evaluation.delivery_note:", evaluation.delivery_note)


def test_stage5_real_llm_chat_and_voice_delivery_flow() -> None:
    answer_text = (
        "JPA N+1 문제는 조회한 엔티티의 연관 엔티티에 접근할 때 "
        "각 엔티티마다 추가 쿼리가 실행되는 문제입니다. "
        "fetch join이나 EntityGraph, batch size 설정으로 해결할 수 있습니다."
    )

    chat_signal, chat_evaluation = evaluate_as_interviewer(
        question_id="q-stage5-chat",
        answer_text=answer_text,
        delivery_metrics=None,
    )
    voice_signal, voice_evaluation = evaluate_as_interviewer(
        question_id="q-stage5-voice",
        answer_text=answer_text,
        delivery_metrics={
            "speech_rate_wpm": 185,
            "filler_count": 7,
        },
    )

    print_result("CHAT RESULT", chat_signal, chat_evaluation)
    print_result("VOICE RESULT", voice_signal, voice_evaluation)

    print("\n===== CONTENT COMPARISON =====")
    print("same_quality:", chat_signal.quality == voice_signal.quality)
    print(
        "accuracy_difference:",
        abs(chat_signal.accuracy - voice_signal.accuracy),
    )
    print(
        "sufficiency_difference:",
        abs(chat_signal.sufficiency - voice_signal.sufficiency),
    )

    assert chat_signal.delivery_note is None
    assert chat_evaluation.delivery_note is None

    assert voice_signal.delivery_note is not None
    assert voice_signal.delivery_note.strip()
    assert voice_evaluation.delivery_note == voice_signal.delivery_note

    delivery_terms = (
        "말하는 속도",
        "발화 속도",
        "빠르",
        "느리",
        "군더더기",
        "필러",
        "추임새",
    )
    content_feedback_terms = (
        "N+1",
        "해결 방법",
        "fetch join",
        "EntityGraph",
        "batch size",
    )
    assert any(
        term in voice_signal.delivery_note
        for term in delivery_terms
    )
    assert not any(
        term in voice_signal.delivery_note
        for term in content_feedback_terms
    )

    assert chat_signal.quality == voice_signal.quality
    assert abs(chat_signal.accuracy - voice_signal.accuracy) <= 0.15
    assert abs(chat_signal.sufficiency - voice_signal.sufficiency) <= 0.15
