"""Assessment 질문 세트 흐름 테스트.

검증 시나리오:

MAIN 답변
→ BONUS_AVAILABLE
→ FOLLOW_UP 질문이 필요하다고 판단

FOLLOW_UP 답변
→ BONUS_AVAILABLE
→ 다시 FOLLOW_UP 질문이 필요하다고 판단

FOLLOW_UP 답변
→ SUFFICIENT
→ 질문 세트 종료
→ 문항별 평가 생성
→ 최종 리포트 생성

확인 내용:
  - current_attempts에 답변 3개가 누적되는지
  - 마지막 답변이 충분해졌을 때 complete_question_set()으로 평가가 생성되는지
  - 문항별 점수(evaluations[0].score)가 생성되는지
  - 최종 평균 점수(report.overall_score)가 생성되는지
  - answer_summary에 세 답변이 모두 포함되는지
"""

import json

from interview.assessment import evaluator
from interview.assessment.agent import AssessmentAgent
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)
from interview.schemas.report import FinalReport
from interview.schemas.signals import AnswerQuality


def make_question(
    question_id: str,
    text: str,
    kind: QuestionKind,
) -> Question:
    """테스트용 질문 객체를 생성한다."""

    return Question(
        question_id=question_id,
        text=text,
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=kind,
        category=QuestionCategory.TECHNICAL,
    )


def test_assessment_two_followups_then_sufficient_report(
    monkeypatch,
) -> None:
    """메인 질문에서 꼬리질문 2번 후 충분한 답변으로 종료되는 흐름을 검증한다."""

    expected_qualities = iter([
        AnswerQuality.BONUS_AVAILABLE,
        AnswerQuality.BONUS_AVAILABLE,
        AnswerQuality.SUFFICIENT,
    ])

    def fake_choice(results):
        """랜덤 평가 결과 대신 원하는 quality를 순서대로 반환한다."""

        expected_quality = next(expected_qualities)

        return next(
            result
            for result in results
            if result.quality == expected_quality
        )

    monkeypatch.setattr(
        evaluator.random,
        "choice",
        fake_choice,
    )

    assessment = AssessmentAgent()

    main_question = make_question(
        question_id="q-main-1",
        text="FastAPI에서 Depends를 사용하는 이유는 무엇인가요?",
        kind=QuestionKind.MAIN,
    )

    follow_up_question_1 = make_question(
        question_id="q-follow-up-1",
        text="Depends를 실제 프로젝트에서 어디에 사용했나요?",
        kind=QuestionKind.FOLLOW_UP,
    )

    follow_up_question_2 = make_question(
        question_id="q-follow-up-2",
        text="Depends를 테스트할 때는 어떻게 처리했나요?",
        kind=QuestionKind.FOLLOW_UP,
    )

    # ============================================================
    # 1. MAIN 답변
    #    평가 결과: BONUS_AVAILABLE
    #    의미: 꼬리질문 필요
    # ============================================================

    signal1 = assessment.evaluate(
        question=main_question,
        answer_text="Depends는 의존성을 주입할 때 사용합니다.",
    )

    assert signal1.quality == AnswerQuality.BONUS_AVAILABLE
    assert len(assessment.current_attempts) == 1
    assert len(assessment.all_attempts) == 1

    print("\n===== 1. MAIN 답변 평가 결과 =====")
    print(
        json.dumps(
            signal1.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    # ============================================================
    # 2. FOLLOW_UP 1 답변
    #    평가 결과: BONUS_AVAILABLE
    #    의미: 아직 추가 꼬리질문 필요
    # ============================================================

    signal2 = assessment.evaluate(
        question=follow_up_question_1,
        answer_text="프로젝트에서는 DB 세션을 주입할 때 사용했습니다.",
    )

    assert signal2.quality == AnswerQuality.BONUS_AVAILABLE
    assert len(assessment.current_attempts) == 2
    assert len(assessment.all_attempts) == 2

    print("\n===== 2. FOLLOW_UP 1 답변 평가 결과 =====")
    print(
        json.dumps(
            signal2.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    # ============================================================
    # 3. FOLLOW_UP 2 답변
    #    평가 결과: SUFFICIENT
    #    의미: 질문 세트 종료 가능
    # ============================================================

    signal3 = assessment.evaluate(
        question=follow_up_question_2,
        answer_text=(
            "테스트에서는 FastAPI의 dependency_overrides를 사용해서 "
            "실제 DB 세션 대신 테스트용 세션으로 교체했습니다."
        ),
    )

    assert signal3.quality == AnswerQuality.SUFFICIENT
    assert len(assessment.current_attempts) == 3
    assert len(assessment.all_attempts) == 3

    print("\n===== 3. FOLLOW_UP 2 답변 평가 결과 =====")
    print(
        json.dumps(
            signal3.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    # ============================================================
    # 4. 질문 세트 종료 및 문항별 평가 생성
    # ============================================================

    assessment.complete_question_set(
        main_question_id=main_question.question_id,
    )

    assert assessment.current_attempts == []
    assert len(assessment.all_attempts) == 3
    assert len(assessment.evaluations) == 1

    evaluation = assessment.evaluations[0]

    assert evaluation.question_id == "q-main-1"
    assert evaluation.topic == "FastAPI"
    assert evaluation.question == "FastAPI에서 Depends를 사용하는 이유는 무엇인가요?"

    # 현재 scoring.py 기준:
    # 파생 질문을 거쳐 마지막에 SUFFICIENT가 되었으므로 80점
    assert evaluation.score == 80.0

    assert evaluation.comment == "초기 답변의 부족한 부분을 파생 질문에서 보완했습니다."

    # answer_summary에는 세 답변이 모두 포함되어야 한다.
    assert "Depends는 의존성을 주입할 때 사용합니다." in evaluation.answer_summary
    assert "프로젝트에서는 DB 세션을 주입할 때 사용했습니다." in evaluation.answer_summary
    assert "dependency_overrides" in evaluation.answer_summary

    assert assessment.competency.topic_scores["FastAPI"] == 80.0

    print("\n===== 4. 문항별 평가 결과 =====")
    print(
        json.dumps(
            evaluation.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    # ============================================================
    # 5. 최종 리포트 생성
    # ============================================================

    report = assessment.finalize()

    assert isinstance(report, FinalReport)

    # 문항이 1개이므로 전체 평균도 80점
    assert report.overall_score == 80.0

    assert report.summary
    assert report.strengths
    assert report.improvement_points
    assert report.learning_recommendations

    assert len(report.evaluations) == 1
    assert report.evaluations[0].score == 80.0
    assert report.evaluations[0].question == evaluation.question
    assert report.evaluations[0].answer_summary == evaluation.answer_summary
    assert report.evaluations[0].comment == evaluation.comment

    print("\n===== 5. 최종 리포트 결과 =====")
    print(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )