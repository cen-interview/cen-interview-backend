"""Assessment fake 구현 테스트."""

from interview.assessment import AssessmentAgent


def test_assessment_evaluates_answer_and_builds_report(sample_question):
    agent = AssessmentAgent()

    signal = agent.evaluate(
        sample_question,
        "JPA N+1 문제는 지연 로딩 상황에서 추가 쿼리가 반복 발생하는 문제입니다.",
    )
    report = agent.finalize()

    assert signal.question_id == sample_question.question_id
    assert signal.quality == "sufficient"
    assert report.evaluations[0].question_id == sample_question.question_id
    assert sample_question.topic in report.strengths
