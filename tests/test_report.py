from interview.assessment.agent import AssessmentAgent
from interview.schemas.question import Difficulty, Question, QuestionKind
from interview.schemas.report import FinalReport


def test_final_report_created_after_question_set():
    assessment = AssessmentAgent()

    main_question = Question(
        question_id="q-1",
        text="FastAPI에서 Depends를 사용하는 이유는?",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=QuestionKind.MAIN,
    )

    follow_up_question = Question(
        question_id="q-2",
        text="FastAPI 답변에서 핵심 개념 부분을 더 설명해 주세요.",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=QuestionKind.FOLLOW_UP,
        parent_question_id="q-1",
    )

    assessment.evaluate(
        question=main_question,
        answer_text="Depends는 의존성 주입입니다.",
    )

    assessment.evaluate(
        question=follow_up_question,
        answer_text="DB 세션 같은 공통 의존성을 주입해서 재사용합니다.",
    )

    assessment.complete_question_set(
        topic="FastAPI",
        main_question_id="q-1",
    )

    report = assessment.finalize()

    assert isinstance(report, FinalReport)
    assert len(report.evaluations) == 1
    assert report.evaluations[0].question_id == "q-1"
    assert report.evaluations[0].topic == "FastAPI"