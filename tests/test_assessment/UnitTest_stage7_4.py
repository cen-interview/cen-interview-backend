"""Assessment 7-4 단위 테스트.

질문 세트가 완료될 때 CompetencyModel.average_score가
누적 AnswerEvaluation 평균으로 갱신되는지 확인한다.
"""

from interview.assessment import agent as assessment_agent_module
from interview.assessment.agent import AssessmentAgent
from interview.assessment.scoring import AnswerAttempt, QuestionSetScore
from interview.schemas.question import Difficulty, QuestionCategory, QuestionKind
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


def make_attempt(
    *,
    question_id: str,
    topic: str,
    answer_text: str = "테스트 답변입니다.",
) -> AnswerAttempt:
    signal = AnswerQualitySignal(
        answer_id=f"answer-{question_id}",
        question_id=question_id,
        quality=AnswerQuality.SUFFICIENT,
        rationale=["단위 테스트용 충분한 답변입니다."],
        accuracy=0.8,
        sufficiency=0.8,
    )

    return AnswerAttempt(
        answer_id=signal.answer_id,
        question_id=question_id,
        question_text=f"{topic} 테스트 질문입니다.",
        question_topic=topic,
        question_kind=QuestionKind.MAIN,
        question_category=QuestionCategory.TECHNICAL,
        question_difficulty=Difficulty.MEDIUM,
        answer_text=answer_text,
        signal=signal,
    )


def test_stage7_4_competency_average_score_updates_after_each_question_set(
    monkeypatch,
) -> None:
    """7-4. 문항 완료마다 평균 점수가 0 고정이 아니라 누적 평균으로 갱신된다."""

    scores = iter(
        [
            QuestionSetScore(
                score=80.0,
                comment="FastAPI 문항 평가 코멘트",
            ),
            QuestionSetScore(
                score=60.0,
                comment="JPA 문항 평가 코멘트",
            ),
        ]
    )

    monkeypatch.setattr(
        assessment_agent_module,
        "score_question_set",
        lambda attempts: next(scores),
    )

    assessment = AssessmentAgent()

    assessment.current_attempts = [
        make_attempt(
            question_id="q-fastapi-1",
            topic="FastAPI",
        )
    ]
    assessment.complete_question_set(
        main_question_id="q-fastapi-1",
    )

    print("\n===== 7-4 첫 번째 문항 완료 후 =====")
    print("topic_scores:", assessment.competency.topic_scores)
    print("average_score:", assessment.competency.average_score)

    assert assessment.competency.topic_scores == {
        "FastAPI": 80.0,
    }
    assert assessment.competency.average_score == 80.0

    assessment.current_attempts = [
        make_attempt(
            question_id="q-jpa-1",
            topic="JPA",
        )
    ]
    assessment.complete_question_set(
        main_question_id="q-jpa-1",
    )

    print("\n===== 7-4 두 번째 문항 완료 후 =====")
    print("topic_scores:", assessment.competency.topic_scores)
    print("average_score:", assessment.competency.average_score)

    assert assessment.competency.topic_scores == {
        "FastAPI": 80.0,
        "JPA": 60.0,
    }
    assert assessment.competency.average_score == 70.0
    assert [evaluation.score for evaluation in assessment.evaluations] == [
        80.0,
        60.0,
    ]
