"""Interviewer와 Assessment의 전체 흐름 테스트.

검증 시나리오:

MAIN 답변
→ MISCONCEPTION
→ CHALLENGE 질문

CHALLENGE 답변
→ BONUS_AVAILABLE
→ FOLLOW_UP 질문

FOLLOW_UP 답변
→ SUFFICIENT
→ 질문 세트 채점
→ 최종 리포트 생성
"""

import json

from interview.assessment import evaluator
from interview.assessment.agent import AssessmentAgent
from interview.interviewer.agent import InterviewerAgent
from interview.interviewer.session import SessionState
from interview.schemas.events import AnswerSubmitted
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionKind,
)
from interview.schemas.report import FinalReport
from interview.schemas.signals import AnswerQuality
from interview.strategy.agent import StrategyAgent


def make_main_question() -> Question:
    """테스트를 시작할 메인 질문을 생성한다."""

    return Question(
        question_id="q-main-1",
        text="FastAPI에서 Depends를 사용하는 이유는 무엇인가요?",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=QuestionKind.MAIN,
    )


def test_main_challenge_follow_up_sufficient_flow(
    monkeypatch,
) -> None:
    """임시 랜덤 평가 결과를 고정해 전체 면접 흐름을 검증한다."""

    # evaluator의 random.choice()가 반환할 quality 순서
    expected_qualities = iter([
        AnswerQuality.MISCONCEPTION,
        AnswerQuality.BONUS_AVAILABLE,
        AnswerQuality.SUFFICIENT,
    ])

    def fake_choice(results):
        """랜덤 선택 대신 원하는 quality를 순서대로 반환한다."""

        expected_quality = next(expected_qualities)

        return next(
            result
            for result in results
            if result.quality == expected_quality
        )

    # evaluator.py가 `import random`을 사용해야 한다.
    monkeypatch.setattr(
        evaluator.random,
        "choice",
        fake_choice,
    )

    main_question = make_main_question()

    # 메인 질문 하나만 평가하면 면접이 종료되도록 설정한다.
    session = SessionState(
        session_id="session-1",
        max_questions=1,
        current_question=main_question,
        asked_count=1,
        main_question_id=main_question.question_id,
        main_topic=main_question.topic,
    )

    assessment = AssessmentAgent()
    strategy = StrategyAgent()

    interviewer = InterviewerAgent(
        session=session,
        strategy=strategy,
        assessment=assessment,
    )

    # ============================================================
    # 1. MAIN 답변
    #    평가 결과: MISCONCEPTION
    #    예상 다음 질문: CHALLENGE
    # ============================================================

    challenge_question = interviewer.handle(
        AnswerSubmitted(
            session_id="session-1",
            question_id=main_question.question_id,
            text=(
                "Depends는 라우터에서 사용할 객체를 "
                "자동으로 생성하는 기능입니다."
            ),
        )
    )

    assert challenge_question is not None
    assert challenge_question.kind == QuestionKind.CHALLENGE
    assert session.current_question == challenge_question

    assert len(assessment.current_attempts) == 1

    main_attempt = assessment.current_attempts[0]

    assert main_attempt.question_id == main_question.question_id
    assert main_attempt.question_kind == QuestionKind.MAIN
    assert (
        main_attempt.signal.quality
        == AnswerQuality.MISCONCEPTION
    )
    assert main_attempt.signal.next_probe_target == (
        "핵심 개념의 정확한 역할"
    )
    assert main_attempt.signal.rationale == [
        "핵심 개념 오해",
        "역할 설명 오류",
    ]

    print("\n===== 1. MAIN 평가 결과 =====")
    print(
        json.dumps(
            main_attempt.signal.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n===== 생성된 CHALLENGE 질문 =====")
    print(
        json.dumps(
            challenge_question.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    # ============================================================
    # 2. CHALLENGE 답변
    #    평가 결과: BONUS_AVAILABLE
    #    예상 다음 질문: FOLLOW_UP
    # ============================================================

    follow_up_question = interviewer.handle(
        AnswerSubmitted(
            session_id="session-1",
            question_id=challenge_question.question_id,
            text=(
                "정정하겠습니다. Depends는 필요한 의존성을 "
                "함수에 주입하는 기능입니다."
            ),
        )
    )

    assert follow_up_question is not None
    assert follow_up_question.kind == QuestionKind.FOLLOW_UP
    assert session.current_question == follow_up_question

    assert len(assessment.current_attempts) == 2

    challenge_attempt = assessment.current_attempts[1]

    assert (
        challenge_attempt.question_id
        == challenge_question.question_id
    )
    assert (
        challenge_attempt.question_kind
        == QuestionKind.CHALLENGE
    )
    assert (
        challenge_attempt.signal.quality
        == AnswerQuality.BONUS_AVAILABLE
    )
    assert challenge_attempt.signal.next_probe_target == (
        "실제 프로젝트 적용 사례"
    )
    assert challenge_attempt.signal.rationale == [
        "기본 개념 설명 확인",
        "실제 적용 사례 부족",
    ]

    print("\n===== 2. CHALLENGE 평가 결과 =====")
    print(
        json.dumps(
            challenge_attempt.signal.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n===== 생성된 FOLLOW_UP 질문 =====")
    print(
        json.dumps(
            follow_up_question.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    # ============================================================
    # 3. FOLLOW_UP 답변
    #    평가 결과: SUFFICIENT
    #    예상 결과: 질문 세트 종료
    # ============================================================

    next_question = interviewer.handle(
        AnswerSubmitted(
            session_id="session-1",
            question_id=follow_up_question.question_id,
            text=(
                "실제 프로젝트에서는 DB 세션과 인증 사용자를 "
                "Depends로 주입했습니다. 테스트에서는 의존성 "
                "오버라이드를 사용해 테스트 객체로 교체했습니다."
            ),
        )
    )

    # max_questions가 1이므로 다음 메인 질문 없이 종료된다.
    assert next_question is None
    assert session.finished is True

    # complete_question_set() 호출 후 현재 질문 세트는 비워진다.
    assert assessment.current_attempts == []

    # 질문 세트 평가 결과가 하나 생성돼야 한다.
    assert len(assessment.evaluations) == 1

    evaluation = assessment.evaluations[0]

    assert evaluation.question_id == "q-main-1"
    assert evaluation.topic == "FastAPI"

    # 마지막 FOLLOW_UP 답변이 SUFFICIENT이므로 최종 상태도 SUFFICIENT
    assert evaluation.quality == AnswerQuality.SUFFICIENT

    # 현재 임시 채점 규칙:
    # 첫 답변은 부족했지만 파생 질문에서 보완 → 80점
    assert evaluation.score == 80.0
    assert evaluation.accuracy == 0.8
    assert evaluation.sufficiency == 0.8

    # 질문 세트에는 답변 세 건이 들어 있어야 한다.
    assert len(evaluation.answer_ids) == 3

    # MAIN을 제외한 CHALLENGE와 FOLLOW_UP 질문 ID
    assert evaluation.derived_question_ids == [
        challenge_question.question_id,
        follow_up_question.question_id,
    ]

    assert (
        assessment.competency.topic_scores["FastAPI"]
        == 80.0
    )

    print("\n===== 3. 질문 세트 평가 결과 =====")
    print(
        json.dumps(
            evaluation.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    # ============================================================
    # 4. 최종 리포트 생성
    # ============================================================

    report = assessment.finalize()

    assert isinstance(report, FinalReport)
    assert report.overall_score == 80.0
    assert len(report.evaluations) == 1
    assert (
        report.evaluations[0].quality
        == AnswerQuality.SUFFICIENT
    )

    print("\n===== 4. 최종 Assessment 결과지 =====")
    print(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )