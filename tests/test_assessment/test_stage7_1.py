"""Assessment 7-1 체크포인트 테스트.

최종 리포트 생성은 문항별 evaluations 전체를 LLM에 한 번 전달하고,
LLM이 생성한 한국어 리포트 본문을 FinalReport에 반영해야 한다.
"""

from interview.assessment import report_builder
from interview.assessment.report_builder import ReportContent
from interview.schemas.report import (
    AnswerEvaluation,
    CompetencyModel,
    QualityTrace,
)


def _print_section(title: str) -> None:
    print(f"\n===== {title} =====")


def _evaluation(
    *,
    question_id: str,
    topic: str,
    question: str,
    answer_summary: str,
    score: float,
    comment: str,
    quality_trace: list[QualityTrace],
) -> AnswerEvaluation:
    return AnswerEvaluation(
        question_id=question_id,
        topic=topic,
        question=question,
        answer_summary=answer_summary,
        score=score,
        comment=comment,
        quality_trace=quality_trace,
    )


def _message_content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))

    return str(getattr(message, "content", message))


def _print_evaluations(evaluations: list[AnswerEvaluation]) -> None:
    for index, evaluation in enumerate(evaluations, start=1):
        print(
            f"[질문 {index}]"
            f"\n- question_id: {evaluation.question_id}"
            f"\n- topic: {evaluation.topic}"
            f"\n- question: {evaluation.question}"
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


def _print_llm_report_result(report) -> None:
    print("[LLM 리포트 결과]")
    print(f"- summary: {report.summary}")
    print(f"- strengths: {report.strengths}")
    print(f"- improvement_points: {report.improvement_points}")
    print(f"- learning_recommendations: {report.learning_recommendations}")


class SpyStructuredReportLLM:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def invoke(self, messages):
        self.calls.append(messages)

        return ReportContent(
            summary=(
                "총 5개 문항에서 JPA 오개념은 압박 질문 이후 정정됐고, "
                "Docker와 Redis는 핵심 개념을 안정적으로 설명했습니다. "
                "FastAPI와 JWT는 개념 방향은 맞았지만 프로젝트 적용 사례와 "
                "보안 흐름 설명을 더 보완할 필요가 있습니다."
            ),
            strengths=[
                "압박 질문 이후 JPA N+1 원인과 해결 방법을 구분해 정정했습니다.",
                "Docker 컨테이너 이미지와 실행 환경의 차이를 실제 배포 맥락에서 설명했습니다.",
                "Redis 캐시 적용 이유와 TTL 전략을 구체적인 장애 방지 관점에서 설명했습니다.",
            ],
            improvement_points=[
                "FastAPI 의존성 주입의 적용 범위를 더 구체적으로 설명해야 합니다.",
                "JWT 인증 흐름에서 access token과 refresh token의 책임을 더 명확히 구분해야 합니다.",
            ],
            learning_recommendations=[
                "오개념이 발생한 주제는 정의, 원인, 해결책 순서로 다시 정리하세요.",
                "인증/인가 주제는 요청 흐름, 토큰 검증, 예외 처리 순서로 말하는 연습을 하세요.",
            ],
        )


class SpyReportLLM:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def with_structured_output(self, schema):
        assert schema is ReportContent
        return SpyStructuredReportLLM(self.calls)


def test_stage7_1_build_report_calls_llm_once_with_all_evaluations(monkeypatch):
    """7-1. evaluations 전체를 입력으로 LLM을 1회 호출해 리포트를 생성한다."""

    llm_calls = []

    def spy_get_llm(temperature: float = 0.3):
        assert temperature <= 0.3
        return SpyReportLLM(llm_calls)

    monkeypatch.setattr(
        report_builder,
        "get_llm",
        spy_get_llm,
        raising=False,
    )

    competency = CompetencyModel(
        topic_scores={
            "JPA": 62.0,
            "FastAPI": 88.0,
            "Docker": 91.0,
            "JWT": 67.0,
            "Redis": 82.0,
        },
        average_score=78.0,
    )
    evaluations = [
        _evaluation(
            question_id="q-jpa-1",
            topic="JPA",
            question="JPA N+1 문제가 왜 발생하고 어떻게 해결할 수 있나요?",
            answer_summary=(
                "처음에는 지연 로딩 자체를 문제 원인으로만 설명했지만, "
                "추가 답변에서 연관 엔티티 접근 시 반복 쿼리가 발생한다는 점과 "
                "fetch join, batch size로 완화할 수 있다는 점을 설명했습니다."
            ),
            score=62.0,
            comment="초기 오개념이 있었지만 challenge 이후 핵심 원인과 해결책을 일부 회복했습니다.",
            quality_trace=[
                QualityTrace(
                    question_kind="main",
                    quality="misconception",
                    target="N+1 발생 원인",
                    rationale=["지연 로딩과 추가 쿼리 발생 원인을 혼동했습니다."],
                ),
                QualityTrace(
                    question_kind="challenge",
                    quality="sufficient",
                    target=None,
                    rationale=["압박 질문 이후 fetch join과 batch size를 구분했습니다."],
                ),
            ],
        ),
        _evaluation(
            question_id="q-fastapi-1",
            topic="FastAPI",
            question="FastAPI에서 Depends를 사용하는 이유와 프로젝트 적용 사례를 설명해 주세요.",
            answer_summary=(
                "Depends가 의존성 주입에 사용된다는 점은 설명했지만, "
                "프로젝트에서 DB 세션이나 인증 사용자 주입에 어떻게 연결되는지는 짧게 답했습니다."
            ),
            score=88.0,
            comment="개념 이해는 좋지만 프로젝트 적용 사례의 구체성이 더 있으면 좋습니다.",
            quality_trace=[
                QualityTrace(
                    question_kind="main",
                    quality="bonus_available",
                    target="Depends 적용 사례",
                    rationale=["개념은 맞지만 프로젝트 적용 사례가 부족했습니다."],
                ),
            ],
        ),
        _evaluation(
            question_id="q-docker-1",
            topic="Docker",
            question="Docker 이미지와 컨테이너의 차이를 배포 관점에서 설명해 주세요.",
            answer_summary=(
                "이미지는 실행 환경을 담은 불변 템플릿이고, 컨테이너는 그 이미지를 실행한 "
                "프로세스라는 점을 배포 재현성과 연결해 설명했습니다."
            ),
            score=91.0,
            comment="핵심 개념과 배포 맥락을 안정적으로 연결했습니다.",
            quality_trace=[
                QualityTrace(
                    question_kind="main",
                    quality="sufficient",
                    target=None,
                    rationale=["이미지와 컨테이너의 차이를 배포 재현성 관점에서 충분히 설명했습니다."],
                ),
            ],
        ),
        _evaluation(
            question_id="q-jwt-1",
            topic="JWT",
            question="JWT 기반 인증에서 access token과 refresh token의 역할을 설명해 주세요.",
            answer_summary=(
                "access token이 API 요청 인증에 쓰인다는 점은 설명했지만, "
                "refresh token의 재발급 흐름과 탈취 시 대응 전략은 충분히 설명하지 못했습니다."
            ),
            score=67.0,
            comment="인증 흐름의 큰 방향은 맞지만 보안 책임 분리가 부족했습니다.",
            quality_trace=[
                QualityTrace(
                    question_kind="main",
                    quality="bonus_available",
                    target="refresh token 재발급 흐름",
                    rationale=["access token 설명은 맞지만 refresh token의 책임과 재발급 흐름이 부족했습니다."],
                ),
                QualityTrace(
                    question_kind="follow_up",
                    quality="bonus_available",
                    target="토큰 탈취 대응",
                    rationale=["추가 질문 이후에도 저장 위치와 폐기 전략 설명이 충분하지 않았습니다."],
                ),
            ],
        ),
        _evaluation(
            question_id="q-redis-1",
            topic="Redis",
            question="Redis 캐시를 사용할 때 TTL을 설정하는 이유를 설명해 주세요.",
            answer_summary=(
                "캐시 데이터가 오래 남아 stale data가 되는 문제를 막고, "
                "장애나 트래픽 급증 상황에서 메모리 사용량을 제어하기 위해 TTL을 둔다고 설명했습니다."
            ),
            score=82.0,
            comment="캐시 무효화와 운영 안정성 관점의 설명이 좋았습니다.",
            quality_trace=[
                QualityTrace(
                    question_kind="main",
                    quality="sufficient",
                    target=None,
                    rationale=["TTL의 목적을 데이터 신선도와 메모리 제어 관점에서 설명했습니다."],
                ),
            ],
        ),
    ]

    _print_section("LLM 입력 질문값")
    print(f"overall_score 예상값: {competency.average_score}")
    print(f"topic_scores: {competency.topic_scores}")
    _print_evaluations(evaluations)

    report = report_builder.build_report(
        competency=competency,
        evaluations=evaluations,
    )

    assert len(llm_calls) == 1

    messages = llm_calls[0]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "최종 리포트" in _message_content(messages[0])

    user_prompt = _message_content(messages[1])
    assert "q-jpa-1" in user_prompt
    assert "q-fastapi-1" in user_prompt
    assert "q-docker-1" in user_prompt
    assert "q-jwt-1" in user_prompt
    assert "q-redis-1" in user_prompt
    assert "JPA" in user_prompt
    assert "FastAPI" in user_prompt
    assert "Docker" in user_prompt
    assert "JWT" in user_prompt
    assert "Redis" in user_prompt
    assert "misconception" in user_prompt
    assert "bonus_available" in user_prompt
    assert "N+1 발생 원인" in user_prompt
    assert "Depends 적용 사례" in user_prompt
    assert "refresh token 재발급 흐름" in user_prompt
    assert "62.0" in user_prompt
    assert "88.0" in user_prompt
    assert "91.0" in user_prompt
    assert "67.0" in user_prompt
    assert "82.0" in user_prompt

    _print_section("LLM 통해 받은 결과")
    _print_llm_report_result(report)

    assert report.overall_score == 78.0
    assert report.summary.startswith("총 5개 문항")
    assert "JPA N+1" in report.strengths[0]
    assert "FastAPI" in report.improvement_points[0]
    assert report.learning_recommendations
    assert report.evaluations == evaluations
