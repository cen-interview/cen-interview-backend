"""Assessment 7-1/7-2/7-3 실제 LLM 흐름 확인 테스트.

이 테스트는 AnswerEvaluation을 직접 만들지 않는다.
질문과 답변 원문만 입력하고 다음 전체 흐름을 확인한다.

1. AssessmentAgent.evaluate()가 답변 평가 LLM을 호출한다.
2. complete_question_set()이 score/comment/quality_trace를 생성한다.
3. finalize()가 evaluations 전체를 최종 리포트 LLM에 전달한다.
"""

from interview.assessment import report_builder
from interview.assessment.agent import AssessmentAgent
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)


def _print_section(title: str) -> None:
    print(f"\n===== {title} =====")


def _question(
    *,
    question_id: str,
    topic: str,
    text: str,
    difficulty: Difficulty = Difficulty.MEDIUM,
) -> Question:
    return Question(
        question_id=question_id,
        text=text,
        topic=topic,
        difficulty=difficulty,
        kind=QuestionKind.MAIN,
        category=QuestionCategory.TECHNICAL,
    )


def _print_question_inputs(cases: list[tuple[Question, str]]) -> None:
    for index, (question, answer_text) in enumerate(cases, start=1):
        print(
            f"[입력 {index}]"
            f"\n- question_id: {question.question_id}"
            f"\n- topic: {question.topic}"
            f"\n- question: {question.text}"
            f"\n- answer_text: {answer_text}"
        )


def _print_evaluations(assessment: AssessmentAgent) -> None:
    for index, evaluation in enumerate(assessment.evaluations, start=1):
        print(
            f"[생성된 평가 {index}]"
            f"\n- question_id: {evaluation.question_id}"
            f"\n- topic: {evaluation.topic}"
            f"\n- answer_summary: {evaluation.answer_summary}"
            f"\n- score: {evaluation.score}"
            f"\n- comment: {evaluation.comment}"
        )

        for trace_index, trace in enumerate(evaluation.quality_trace, start=1):
            print(
                f"  quality_trace {trace_index}: "
                f"kind={trace.question_kind}, "
                f"quality={trace.quality}, "
                f"target={trace.target}, "
                f"rationale={trace.rationale}"
            )


def _print_report(report) -> None:
    print("[최종 LLM 리포트]")
    print(f"- overall_score: {report.overall_score}")
    print(f"- summary: {report.summary}")
    print(f"- strengths: {report.strengths}")
    print(f"- improvement_points: {report.improvement_points}")
    print(f"- learning_recommendations: {report.learning_recommendations}")


def _find_prompt_line(prompt: str, prefix: str) -> str:
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line

    raise AssertionError(f"{prefix} 줄을 찾지 못했습니다.")


def test_stage7_1_real_llm_from_questions_and_answers_only(monkeypatch):
    """질문/답변만 입력해 평가와 최종 리포트를 모두 실제 LLM 경로로 생성한다."""

    def fail_if_report_fallback_is_used(*args, **kwargs):
        raise AssertionError("최종 리포트 LLM 호출에 실패해 fallback 리포트가 사용되었습니다.")

    monkeypatch.setattr(
        report_builder,
        "_temporary_report_content",
        fail_if_report_fallback_is_used,
    )
    cases = [
        (
            _question(
                question_id="q-jpa-1",
                topic="JPA",
                text="JPA N+1 문제가 왜 발생하고 어떻게 해결할 수 있나요?",
            ),
            (
                "N+1은 지연 로딩 연관 엔티티를 반복해서 접근할 때 추가 쿼리가 "
                "여러 번 나가는 문제입니다. fetch join이나 EntityGraph, batch size로 "
                "조회 전략을 조정해서 줄일 수 있습니다."
            ),
        ),
        (
            _question(
                question_id="q-fastapi-1",
                topic="FastAPI",
                text="FastAPI에서 Depends를 사용하는 이유와 프로젝트 적용 사례를 설명해 주세요.",
            ),
            (
                "Depends는 의존성 주입을 위해 사용합니다. 프로젝트에서는 DB 세션을 "
                "요청마다 주입하거나 현재 로그인 사용자를 가져오는 데 사용할 수 있습니다."
            ),
        ),
        (
            _question(
                question_id="q-docker-1",
                topic="Docker",
                text="Docker 이미지와 컨테이너의 차이를 배포 관점에서 설명해 주세요.",
            ),
            (
                "이미지는 애플리케이션 실행에 필요한 파일과 설정을 담은 템플릿이고, "
                "컨테이너는 그 이미지를 실제로 실행한 프로세스입니다. 그래서 같은 이미지를 "
                "사용하면 서버가 바뀌어도 비슷한 실행 환경을 재현할 수 있습니다."
            ),
        ),
        (
            _question(
                question_id="q-jwt-1",
                topic="JWT",
                text="JWT 기반 인증에서 access token과 refresh token의 역할을 설명해 주세요.",
            ),
            (
                "access token은 API 요청 때 인증 정보를 전달하는 짧은 수명의 토큰입니다. "
                "refresh token은 access token이 만료됐을 때 재발급을 받기 위한 토큰이고, "
                "탈취 위험이 있으니 저장 위치와 폐기 정책을 신경 써야 합니다."
            ),
        ),
        (
            _question(
                question_id="q-redis-1",
                topic="Redis",
                text="Redis 캐시를 사용할 때 TTL을 설정하는 이유를 설명해 주세요.",
            ),
            (
                "TTL은 캐시 데이터가 너무 오래 남아 stale data가 되는 것을 막고, "
                "메모리를 계속 점유하는 문제를 줄이기 위해 설정합니다. 트래픽이 몰릴 때도 "
                "캐시 만료 정책이 있어야 운영 안정성을 유지하기 쉽습니다."
            ),
        ),
    ]

    _print_section("입력: 질문 + 답변 원문만")
    _print_question_inputs(cases)

    assessment = AssessmentAgent()

    _print_section("평가 LLM 호출 및 질문 세트 완료")
    for question, answer_text in cases:
        signal = assessment.evaluate(
            question=question,
            answer_text=answer_text,
        )
        print(
            f"- {question.question_id}: "
            f"quality={signal.quality.value}, "
            f"accuracy={signal.accuracy}, "
            f"sufficiency={signal.sufficiency}, "
            f"target={signal.next_probe_target}"
        )
        assessment.complete_question_set(
            main_question_id=question.question_id,
        )

    _print_section("생성된 evaluations")
    _print_evaluations(assessment)

    user_prompt = report_builder._build_report_user_prompt(
        competency=assessment.competency,
        evaluations=assessment.evaluations,
        overall_score=report_builder._calculate_overall_score(
            assessment.evaluations
        ),
    )
    topics_to_improve_line = _find_prompt_line(
        user_prompt,
        "topics_to_improve:",
    )

    _print_section("7-3 개선 우선순위")
    print(topics_to_improve_line)

    report = assessment.finalize()

    _print_section("최종 리포트 LLM 결과")
    _print_report(report)

    assert len(assessment.evaluations) == len(cases)
    assert all(evaluation.quality_trace for evaluation in assessment.evaluations)
    assert report.summary
    assert report.strengths
    assert report.improvement_points
    assert report.learning_recommendations
    assert report.evaluations == assessment.evaluations
