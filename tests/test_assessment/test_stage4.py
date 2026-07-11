from interview.assessment import graph as assessment_graph
from interview.assessment.evaluator import JudgeResult
from interview.assessment.graph import AssessmentState
from interview.assessment.scoring import AnswerAttempt
from interview.schemas.evidence import EvidenceChunk, SourceType
from interview.schemas.question import (
    Difficulty,
    Question,
    QuestionCategory,
    QuestionKind,
)
from interview.schemas.signals import (
    AnswerQuality,
    AnswerQualitySignal,
    ConflictType,
)


class FakeStructuredLLM:
    def __init__(self, result: JudgeResult) -> None:
        self.result = result
        self.messages: list[dict[str, str]] = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages: list[dict[str, str]]) -> JudgeResult:
        self.messages = messages
        return self.result


def print_execution_process(
    *,
    label: str,
    answer_text: str,
    comparison_source: str,
    executed_nodes: list[str],
    state: AssessmentState,
) -> None:
    print(f"\n===== {label} =====")
    print("[1. 현재 답변]")
    print(answer_text)
    print("\n[2. 비교 근거]")
    print(comparison_source)
    print("\n[3. 실행 노드]")
    print(" -> ".join(executed_nodes))
    print("\n[4. conflict_check 결과]")
    print("quality:", state.judge_result.quality)
    print("conflict_type:", state.judge_result.conflict_type)
    print("conflict_suspected:", state.judge_result.conflict_suspected)
    print("next_probe_target:", state.judge_result.next_probe_target)
    print("rationale:", state.judge_result.rationale)
    print("\n[5. 최종 신호]")
    print(state.final_signal)


def make_question(
    *,
    question_id: str,
    text: str,
    topic: str,
    category: QuestionCategory,
) -> Question:
    return Question(
        question_id=question_id,
        text=text,
        topic=topic,
        difficulty=Difficulty.MEDIUM,
        kind=QuestionKind.MAIN,
        category=category,
    )


def make_previous_attempt() -> AnswerAttempt:
    signal = AnswerQualitySignal(
        answer_id="answer-n-plus-one",
        question_id="q-n-plus-one",
        quality=AnswerQuality.SUFFICIENT,
        rationale=["지연 로딩 시 연관 엔티티 접근으로 추가 쿼리가 발생한다고 설명했습니다."],
        accuracy=0.9,
        sufficiency=0.8,
    )

    return AnswerAttempt(
        answer_id=signal.answer_id,
        question_id="q-n-plus-one",
        question_text="JPA N+1 문제가 발생하는 원인을 설명해 주세요.",
        question_topic="JPA 지연 로딩",
        question_kind=QuestionKind.MAIN,
        question_category=QuestionCategory.TECHNICAL,
        question_difficulty=Difficulty.MEDIUM,
        answer_text=(
            "지연 로딩된 연관 엔티티에 접근할 때 엔티티마다 추가 쿼리가 "
            "실행되면서 N+1 문제가 발생합니다."
        ),
        signal=signal,
    )


def test_stage4_2_evidence_conflict_is_recorded_separately(monkeypatch) -> None:
    question = make_question(
        question_id="q-project-storage",
        text="프로젝트에서 사용자 세션을 어디에 저장했나요?",
        topic="세션 저장소",
        category=QuestionCategory.PROJECT,
    )
    evidence = EvidenceChunk(
        chunk_id="chunk-session-table",
        text="사용자 세션은 MySQL의 session 테이블에 저장한다.",
        source_type=SourceType.GITHUB,
        source_url="https://github.com/example/project/session.py",
        topic="세션 저장소",
        confidence=0.95,
    )
    conflict_result = JudgeResult(
        quality=AnswerQuality.CONFIRM_NEGATIVE,
        next_probe_target="Redis와 MySQL session 테이블 중 실제 저장소",
        rationale=[
            "[Evidence 충돌] 현재 답변은 Redis 저장을 주장하지만 "
            "Evidence에는 MySQL session 테이블에 저장한다고 기록되어 있습니다."
        ],
        conflict_type=ConflictType.EVIDENCE_CONFLICT,
        conflict_suspected=True,
        accuracy=0.2,
        sufficiency=0.7,
    )
    fake_llm = FakeStructuredLLM(conflict_result)
    monkeypatch.setattr(assessment_graph, "get_llm", lambda **_kwargs: fake_llm)

    state = AssessmentState(
        question=question,
        answer_text="사용자 세션은 Redis에 저장했습니다.",
        evidence_chunks=[evidence],
        judge_result=JudgeResult(
            quality=AnswerQuality.CONFIRM_NEGATIVE,
            next_probe_target="세션 저장소",
            rationale=["Evidence와 충돌할 가능성이 있습니다."],
            conflict_suspected=True,
            accuracy=0.4,
            sufficiency=0.7,
        ),
    )

    executed_nodes = []
    executed_nodes.append("conflict_check")
    state = assessment_graph.conflict_check(state)
    executed_nodes.append("finalize_signal")
    state = assessment_graph.finalize_signal(state)

    print_execution_process(
        label="SCENARIO 3: EVIDENCE CONFLICT",
        answer_text=state.answer_text,
        comparison_source=evidence.text,
        executed_nodes=executed_nodes,
        state=state,
    )

    user_prompt = fake_llm.messages[1]["content"]
    assert "사용자 세션은 Redis에 저장했습니다." in user_prompt
    assert "MySQL의 session 테이블" in user_prompt
    assert state.final_signal is not None
    assert state.final_signal.quality == AnswerQuality.CONFIRM_NEGATIVE
    assert state.final_signal.conflict_type == ConflictType.EVIDENCE_CONFLICT
    assert state.final_signal.rationale[0].startswith("[Evidence 충돌]")


def test_stage4_2_self_contradiction_is_recorded_separately(monkeypatch) -> None:
    question = make_question(
        question_id="q-lazy-loading",
        text="지연 로딩은 즉시 로딩보다 항상 유리한가요?",
        topic="JPA 지연 로딩",
        category=QuestionCategory.TECHNICAL,
    )
    conflict_result = JudgeResult(
        quality=AnswerQuality.MISCONCEPTION,
        next_probe_target="지연 로딩의 장단점과 N+1 발생 조건",
        rationale=[
            "[자기모순] 이전에는 지연 로딩으로 N+1이 발생할 수 있다고 "
            "설명했지만, 현재는 지연 로딩이 항상 유리하다고 단정했습니다."
        ],
        conflict_type=ConflictType.SELF_CONTRADICTION,
        conflict_suspected=True,
        accuracy=0.3,
        sufficiency=0.6,
    )
    fake_llm = FakeStructuredLLM(conflict_result)
    monkeypatch.setattr(assessment_graph, "get_llm", lambda **_kwargs: fake_llm)
    monkeypatch.setattr(
        assessment_graph.evaluator,
        "_judge_with_llm",
        lambda **_kwargs: JudgeResult(
            quality=AnswerQuality.MISCONCEPTION,
            next_probe_target="지연 로딩의 장단점",
            rationale=["과도한 일반화이며 이전 설명과 충돌할 수 있습니다."],
            conflict_suspected=True,
            accuracy=0.3,
            sufficiency=0.6,
        ),
    )

    state = AssessmentState(
        question=question,
        answer_text="지연 로딩은 쿼리 수를 줄이므로 즉시 로딩보다 항상 유리합니다.",
        history=[make_previous_attempt()],
    )

    executed_nodes = []
    executed_nodes.append("judge")
    state = assessment_graph.judge(state)
    executed_nodes.append("conflict_check")
    state = assessment_graph.conflict_check(state)
    executed_nodes.append("finalize_signal")
    state = assessment_graph.finalize_signal(state)

    print_execution_process(
        label="SCENARIO 4: SELF CONTRADICTION",
        answer_text=state.answer_text,
        comparison_source=state.same_topic_history_summary,
        executed_nodes=executed_nodes,
        state=state,
    )

    user_prompt = fake_llm.messages[1]["content"]
    assert "N+1 문제가 발생합니다" in user_prompt
    assert "항상 유리합니다" in user_prompt
    assert state.final_signal is not None
    assert state.final_signal.quality == AnswerQuality.MISCONCEPTION
    assert state.final_signal.conflict_type == ConflictType.SELF_CONTRADICTION
    assert state.final_signal.rationale[0].startswith("[자기모순]")
