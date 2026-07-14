"""Assessment 1단계 체크포인트 테스트.

이 파일은 전체 플로우보다 1단계 설계 결정이 코드에서 지켜지는지 확인한다.
"""

import pytest

from interview.assessment import evaluator
from interview.assessment.agent import AssessmentAgent
from interview.assessment.evaluator import JudgeResult
from interview.schemas.evidence import EvidenceChunk, SourceType
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)
from interview.schemas.signals import AnswerQuality


def make_question(
    question_id: str,
    text: str,
    category: QuestionCategory,
    kind: QuestionKind = QuestionKind.MAIN,
) -> Question:
    """체크포인트 테스트용 질문 객체를 생성한다."""

    return Question(
        question_id=question_id,
        text=text,
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=kind,
        category=category,
    )


def test_stage1_1_project_question_uses_evidence(monkeypatch) -> None:
    """1-1. PROJECT 질문은 Evidence를 조회해서 judge에 넘긴다."""

    searched = False

    def fake_search_evidence(query, topic):
        nonlocal searched
        searched = True

        assert query == "프로젝트에서 Depends를 어디에 적용했나요?"
        assert topic == "FastAPI"

        return [
            EvidenceChunk(
                chunk_id="chunk-project-1",
                text="프로젝트에서 DB 세션 주입에 Depends를 사용했다.",
                source_type=SourceType.GITHUB,
                source_url="https://example.com/repo",
                topic="FastAPI",
                confidence=0.8,
            )
        ]

    def fake_judge_with_llm(**kwargs):
        evidence_chunks = kwargs["evidence_chunks"]

        assert len(evidence_chunks) == 1
        assert evidence_chunks[0].chunk_id == "chunk-project-1"

        return JudgeResult(
            quality=AnswerQuality.SUFFICIENT,
            rationale=["프로젝트 Evidence를 확인했습니다."],
        )

    monkeypatch.setattr(
        evaluator,
        "search_evidence",
        fake_search_evidence,
    )
    monkeypatch.setattr(
        evaluator,
        "_judge_with_llm",
        fake_judge_with_llm,
    )

    signal = evaluator.judge_answer(
        question=make_question(
            question_id="q-project-1",
            text="프로젝트에서 Depends를 어디에 적용했나요?",
            category=QuestionCategory.PROJECT,
        ),
        answer_text="DB 세션 주입에 사용했습니다.",
    )

    assert searched is True
    assert signal.quality == AnswerQuality.SUFFICIENT


def test_stage1_1_technical_question_does_not_use_evidence(monkeypatch) -> None:
    """1-1. TECHNICAL 질문은 Evidence를 조회하지 않고 자체 판단으로 judge에 들어간다."""

    def fail_search_evidence(*args, **kwargs):
        raise AssertionError("TECHNICAL 질문은 Evidence를 조회하지 않아야 합니다.")

    def fake_judge_with_llm(**kwargs):
        assert kwargs["evidence_chunks"] == []

        return JudgeResult(
            quality=AnswerQuality.SUFFICIENT,
            rationale=["기술개념은 Evidence 없이 판단했습니다."],
        )

    monkeypatch.setattr(
        evaluator,
        "search_evidence",
        fail_search_evidence,
    )
    monkeypatch.setattr(
        evaluator,
        "_judge_with_llm",
        fake_judge_with_llm,
    )

    signal = evaluator.judge_answer(
        question=make_question(
            question_id="q-technical-1",
            text="Depends의 개념을 설명해 주세요.",
            category=QuestionCategory.TECHNICAL,
        ),
        answer_text="Depends는 의존성 주입에 사용합니다.",
    )

    assert signal.quality == AnswerQuality.SUFFICIENT


def test_stage1_2_all_attempts_survives_question_set_completion(monkeypatch) -> None:
    """1-2. current_attempts는 세트 종료 시 비워지고 all_attempts는 전체 이력을 유지한다."""

    monkeypatch.setattr(
        evaluator,
        "search_evidence",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        evaluator,
        "_judge_with_llm",
        lambda **kwargs: JudgeResult(
            quality=AnswerQuality.SUFFICIENT,
            rationale=["충분한 답변입니다."],
        ),
    )

    assessment = AssessmentAgent()
    question = make_question(
        question_id="q-main-1",
        text="Depends의 개념을 설명해 주세요.",
        category=QuestionCategory.TECHNICAL,
    )

    assessment.evaluate(
        question=question,
        answer_text="Depends는 의존성 주입에 사용합니다.",
    )

    assert len(assessment.current_attempts) == 1
    assert len(assessment.all_attempts) == 1

    assessment.complete_question_set(
        main_question_id=question.question_id,
    )

    assert assessment.current_attempts == []
    assert len(assessment.all_attempts) == 1
    assert assessment.all_attempts[0].answer_text == "Depends는 의존성 주입에 사용합니다."


def test_stage1_3_conflict_check_runs_only_when_suspected(monkeypatch) -> None:
    """1-3. 1차 judge가 충돌을 의심할 때만 정밀 충돌 검사를 실행한다."""

    conflict_check_calls = 0

    def fake_run_conflict_check(**kwargs):
        nonlocal conflict_check_calls
        conflict_check_calls += 1

        return kwargs["fallback_result"].model_copy(
            update={"conflict_suspected": False}
        )

    monkeypatch.setattr(
        evaluator,
        "search_evidence",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        evaluator,
        "_run_conflict_check",
        fake_run_conflict_check,
    )

    monkeypatch.setattr(
        evaluator,
        "_judge_with_llm",
        lambda **kwargs: JudgeResult(
            quality=AnswerQuality.BONUS_AVAILABLE,
            next_probe_target="프로젝트 적용 사례",
            rationale=["추가 확인이 필요합니다."],
            conflict_suspected=False,
        ),
    )

    evaluator.judge_answer(
        question=make_question(
            question_id="q-no-conflict-check",
            text="Depends의 개념을 설명해 주세요.",
            category=QuestionCategory.TECHNICAL,
        ),
        answer_text="Depends는 의존성 주입에 사용합니다.",
    )

    assert conflict_check_calls == 0

    monkeypatch.setattr(
        evaluator,
        "_judge_with_llm",
        lambda **kwargs: JudgeResult(
            quality=AnswerQuality.BONUS_AVAILABLE,
            next_probe_target="프로젝트 적용 사례",
            rationale=["충돌 의심이 있어 정밀 확인이 필요합니다."],
            conflict_suspected=True,
        ),
    )

    signal = evaluator.judge_answer(
        question=make_question(
            question_id="q-conflict-check",
            text="Depends의 프로젝트 적용 사례를 설명해 주세요.",
            category=QuestionCategory.TECHNICAL,
        ),
        answer_text="테스트에서는 dependency_overrides를 사용했습니다.",
    )

    assert conflict_check_calls == 1
    assert signal.quality == AnswerQuality.BONUS_AVAILABLE


def test_stage1_3_conflict_check_preserves_fallback_when_no_conflict() -> None:
    """1-3. 정밀 검사에서 충돌이 없으면 기존 1차 judge 결과를 유지한다."""

    fallback_result = JudgeResult(
        quality=AnswerQuality.BONUS_AVAILABLE,
        next_probe_target="테스트 적용 사례",
        rationale=["답변은 맞지만 적용 사례가 부족합니다."],
        conflict_suspected=True,
    )

    result = evaluator._run_conflict_check(
        question=make_question(
            question_id="q-conflict-fallback",
            text="프로젝트 적용 사례를 더 설명해 주세요.",
            category=QuestionCategory.TECHNICAL,
            kind=QuestionKind.FOLLOW_UP,
        ),
        answer_text="테스트에서는 dependency_overrides를 사용했습니다.",
        evidence_chunks=[],
        history=[],
        fallback_result=fallback_result,
    )

    assert result.quality == AnswerQuality.BONUS_AVAILABLE
    assert result.next_probe_target == "테스트 적용 사례"
    assert result.conflict_suspected is False


def test_stage1_3_history_summary_contains_previous_attempts(monkeypatch) -> None:
    """1-3. 이전 답변 이력은 judge에서 사용할 수 있는 요약 문자열로 변환된다."""

    monkeypatch.setattr(
        evaluator,
        "search_evidence",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        evaluator,
        "_judge_with_llm",
        lambda **kwargs: JudgeResult(
            quality=AnswerQuality.SUFFICIENT,
            rationale=["충분한 답변입니다."],
        ),
    )

    assessment = AssessmentAgent()
    first_question = make_question(
        question_id="q-history-main",
        text="Depends의 개념을 설명해 주세요.",
        category=QuestionCategory.TECHNICAL,
    )

    assessment.evaluate(
        question=first_question,
        answer_text="Depends는 의존성 주입에 사용합니다.",
    )

    history_summary = evaluator._build_history_summary(assessment.all_attempts)

    assert "FastAPI" in history_summary
    assert "Depends는 의존성 주입에 사용합니다." in history_summary


@pytest.mark.xfail(
    reason="1-4는 아직 실제 LLM 모델 설정/타임아웃 정책이 코드로 확정되지 않았습니다.",
)
def test_stage1_4_judge_model_policy_is_configurable() -> None:
    """1-4. judge용 고정밀 모델과 지연 허용선은 설정값으로 검증 가능해야 한다."""

    from interview.config import settings

    assert settings.assessment_judge_model
    assert settings.assessment_timeout_seconds <= 3.0
