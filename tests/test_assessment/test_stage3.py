"""Assessment 3단계 그래프 경로 테스트.

3-5 체크포인트:
- 충돌 의심이 없으면 conflict_check 노드를 건너뛴다.
- 충돌 의심이 있으면 conflict_check 노드를 거쳐 finalize_signal로 간다.
"""

from interview.assessment import graph as assessment_graph
from interview.assessment.evaluator import JudgeResult
from interview.assessment.graph import AssessmentState
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)
from interview.schemas.signals import AnswerQuality


def _question(question_id: str = "q-stage3") -> Question:
    return Question(
        question_id=question_id,
        text="트랜잭션 전파 옵션을 설명해주세요.",
        topic="Spring Transaction",
        difficulty=Difficulty.MEDIUM,
        kind=QuestionKind.MAIN,
        category=QuestionCategory.TECHNICAL,
    )


def _judge_result(
    *,
    quality: AnswerQuality = AnswerQuality.SUFFICIENT,
    conflict_suspected: bool = False,
) -> JudgeResult:
    return JudgeResult(
        quality=quality,
        next_probe_target=None,
        rationale=["stage3 테스트용 판정입니다."],
        conflict_suspected=conflict_suspected,
        accuracy=0.8,
        sufficiency=0.75,
    )


def _run_nodes_like_graph(state: AssessmentState) -> tuple[AssessmentState, list[str]]:
    """LangGraph가 실행할 노드 순서를 테스트에서 명시적으로 따라간다."""

    executed_nodes = []

    executed_nodes.append("retrieve_evidence")
    state = assessment_graph.retrieve_evidence(state)

    executed_nodes.append("judge")
    state = assessment_graph.judge(state)

    next_node = assessment_graph.route_after_judge(state)
    if next_node == "conflict_check":
        executed_nodes.append("conflict_check")
        state = assessment_graph.conflict_check(state)

    executed_nodes.append("finalize_signal")
    state = assessment_graph.finalize_signal(state)

    return state, executed_nodes


def _print_stage3_result(
    label: str,
    state: AssessmentState,
    executed_nodes: list[str],
) -> None:
    print(f"\n===== {label} =====")
    print("executed_nodes:", executed_nodes)
    print("judge_result:", state.judge_result)
    print("final_signal:", state.final_signal)


def test_stage3_5_no_conflict_path_skips_conflict_check(monkeypatch):
    monkeypatch.setattr(
        assessment_graph.evaluator,
        "_judge_with_llm",
        lambda **kwargs: _judge_result(conflict_suspected=False),
    )

    state = AssessmentState(
        question=_question("q-no-conflict"),
        answer_text="REQUIRED는 기존 트랜잭션이 있으면 참여하고 없으면 새로 만듭니다.",
    )

    result_state, executed_nodes = _run_nodes_like_graph(state)

    _print_stage3_result("NO CONFLICT PATH", result_state, executed_nodes)

    assert executed_nodes == [
        "retrieve_evidence",
        "judge",
        "finalize_signal",
    ]
    assert result_state.final_signal is not None
    assert result_state.final_signal.question_id == "q-no-conflict"
    assert result_state.final_signal.quality == AnswerQuality.SUFFICIENT


def test_stage3_5_conflict_path_runs_conflict_check_before_finalize(monkeypatch):
    monkeypatch.setattr(
        assessment_graph.evaluator,
        "_judge_with_llm",
        lambda **kwargs: _judge_result(
            quality=AnswerQuality.BONUS_AVAILABLE,
            conflict_suspected=True,
        ),
    )

    state = AssessmentState(
        question=_question("q-conflict"),
        answer_text="이전 답변과 충돌할 수 있는 답변입니다.",
    )

    result_state, executed_nodes = _run_nodes_like_graph(state)

    _print_stage3_result("CONFLICT PATH", result_state, executed_nodes)

    assert executed_nodes == [
        "retrieve_evidence",
        "judge",
        "conflict_check",
        "finalize_signal",
    ]
    assert result_state.judge_result is not None
    assert result_state.judge_result.conflict_suspected is False
    assert result_state.final_signal is not None
    assert result_state.final_signal.question_id == "q-conflict"
    assert result_state.final_signal.quality == AnswerQuality.BONUS_AVAILABLE


if __name__ == "__main__":
    from unittest.mock import patch

    with patch.object(
        assessment_graph.evaluator,
        "_judge_with_llm",
        lambda **kwargs: _judge_result(conflict_suspected=False),
    ):
        no_conflict_state = AssessmentState(
            question=_question("q-no-conflict"),
            answer_text="REQUIRED는 기존 트랜잭션이 있으면 참여하고 없으면 새로 만듭니다.",
        )
        result_state, executed_nodes = _run_nodes_like_graph(no_conflict_state)
        _print_stage3_result("NO CONFLICT PATH", result_state, executed_nodes)

    with patch.object(
        assessment_graph.evaluator,
        "_judge_with_llm",
        lambda **kwargs: _judge_result(
            quality=AnswerQuality.BONUS_AVAILABLE,
            conflict_suspected=True,
        ),
    ):
        conflict_state = AssessmentState(
            question=_question("q-conflict"),
            answer_text="이전 답변과 충돌할 수 있는 답변입니다.",
        )
        result_state, executed_nodes = _run_nodes_like_graph(conflict_state)
        _print_stage3_result("CONFLICT PATH", result_state, executed_nodes)
