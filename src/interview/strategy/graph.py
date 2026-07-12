"""메인 질문 생성 그래프 (LangGraph).

next_question() 내부의 pick_topic → retrieve_evidence → generate → validate
흐름을 명시적 그래프로 표현한다. 근거 부족 시 대체 주제로 재시도하고,
검증 실패 시 최대 1회 재생성한다.

파생 질문(follow_up/challenge/confirm/trap/hint)은 분기/재시도가 없어
그래프 없이 question_gen.py에서 단일 LLM 호출로 유지한다.

주의 - StrategyState와 QuestionGenState는 다른 클래스:
    StrategyState (state.py)
        면접 세션 전체의 출제 이력. 한 면접 세션이 끝날 때까지 유지된다.

    QuestionGenState (이 파일)
        메인 질문 "한 개"를 생성하는 동안만 쓰는 임시 작업 상태.
        next_question() 호출 1회마다 새로 만들어지고, 끝나면 버려진다.
        내부에 strategy_state 필드로 StrategyState를 참조해서 읽기만 한다.
"""
import random
from uuid import uuid4

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from interview.evidence.retrieval import search_evidence
from interview.llm.client import get_llm
from interview.schemas.evidence import CoverageMap, EvidenceChunk, TopicCoverage
from interview.schemas.question import Difficulty, Question, QuestionKind
from interview.strategy.prompts import QUESTION_GEN_SYSTEM
from interview.strategy.question_gen import GeneratedQuestion
from interview.strategy.state import StrategyState

from interview.strategy.personalization import get_weak_topics # stub

# 주제 선택 시 confidence 상위 몇 개를 후보 풀로 삼을지 (agent.py에서 그대로 가져옴)
_TOP_N_POOL = 3

# 근거 청크 중 신뢰도(confidence)가 이 값 미만이면 프롬프트/evidence_ids에서 제외
_EVIDENCE_CONFIDENCE_THRESHOLD = 0.4

# 근거 부족으로 pick_topic을 다시 시도하는 최대 횟수 (무한 루프 방지)
_MAX_RETRY = 3

# validate 실패 시 generate를 다시 시도하는 최대 횟수
_MAX_REGENERATE = 1

# "면접 초반"으로 간주할 메인 질문 수
_EARLY_QUESTION_THRESHOLD = 3  

class QuestionGenState(BaseModel):
    """질문 생성 그래프의 상태. (StrategyState와는 다른 클래스 - 위 모듈 docstring 참고)

    Attributes:
        coverage:
            주제 선정에 참고하는 커버리지 맵. pick_topic이 읽는다.

        strategy_state:
            출제 이력. pick_topic(중복/직전주제 회피), generate(중복 질문
            회피)에서 읽는다. StrategyAgent.state를 그대로 참조한다.

        difficulty:
            생성할 질문의 목표 난이도. 그래프 진입 시 확정되어 있다.
        
        user_id:
            면접 응시자 식별자. 아직 API 계층에서 실제 값이
            흘러들어오지 않아 현재는 None으로 동작.

        topic:
            현재 시도 중인 주제. pick_topic 노드에서 채워진다.

        tried_topics:
            이미 시도했다가 근거 부족으로 대체된 주제 목록. pick_topic이
            같은 주제를 다시 고르지 않기 위해 참고한다.

        evidence_chunks:
            retrieve_evidence 노드가 조회한 근거 청크 목록.
        
        weak_history_topics:
            이전 면접 이력에서 약점으로 판단된 주제 목록 (get_weak_topics로 조회).
            question_count가 _EARLY_QUESTION_THRESHOLD 미만인 초반 구간에서
            pick_topic이 우선 배치 대상으로 사용한다. 이력이 없으면 빈 리스트.

        retry_count:
            대체 주제 재시도 횟수. 무한 루프 방지를 위한 상한 판단에 쓴다.

        generated_text:
            generate 노드가 만든 질문 문장 (검증 전).

        generated_category:
            generate 노드가 만든 질문 카테고리.

        validation_failed:
            validate 노드에서 검증에 실패했는지 여부.

        validation_reason:
            검증 실패 이유 (예: "이미 한 질문과 유사함"). 재생성 시
            generate 노드가 이 이유를 프롬프트에 반영해 같은 실수를
            반복하지 않도록 한다.

        regenerate_count:
            재생성 시도 횟수. 최대 1회로 제한한다 (retry_count와 동일한
            카운터 패턴).

        result:
            그래프 최종 산출물. END에 도달하면 채워진다.
    """

    coverage: CoverageMap = Field(default_factory=CoverageMap)
    strategy_state: StrategyState = Field(default_factory=StrategyState)
    difficulty: Difficulty = Difficulty.EASY
    user_id: str | None = None

    topic: str | None = None
    tried_topics: list[str] = Field(default_factory=list)
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    weak_history_topics: list[str] = Field(default_factory=list)
    retry_count: int = 0

    generated_text: str | None = None
    generated_category: str | None = None
    validation_failed: bool = False
    validation_reason: str | None = None
    regenerate_count: int = 0

    result: Question | None = None

def pick_topic(state: QuestionGenState) -> dict:
    """다음 시도할 주제를 고른다.
    
            선택 순서:
            1) 근거가 약한 주제(weak_topics)는 후보에서 제외한다.
            2) 남은 후보 중 아직 묻지 않은 주제를 우선한다.
            3) 직전 주제와 연속되지 않게 회피한다.
            4) 모든 후보를 다 물었다면(주제 소진) 처음부터 다시 순환한다.
            5) 근거가 있는 주제가 하나도 없으면(coverage 미주입 등) 폴백 주제를
            반환한다.

        남은 후보를 confidence 높은 순으로 정렬한 뒤, 상위 _TOP_N_POOL개 안에서
        무작위로 하나를 선택한다 (confidence 우선 + 매번 다른 순서 확보).
    """
    updates: dict = {}

    # 재시도로 들어온 경우: 방금 실패한 topic을 기록하고 카운터 증가
    if state.topic is not None:
        updates["tried_topics"] = state.tried_topics + [state.topic]
        updates["retry_count"] = state.retry_count + 1

    all_topics = list(state.coverage.topic_coverage.keys())
    tried = set(updates.get("tried_topics", state.tried_topics))

    if not all_topics:
        updates["topic"] = "FastAPI"
        return updates

    weak = set(state.coverage.weak_topics())
    candidates = [t for t in all_topics if t not in weak and t not in tried]
    if not candidates:
        candidates = [t for t in all_topics if t not in tried]
    if not candidates:
        candidates = all_topics

    asked = state.strategy_state.topic_counts()
    unasked = [t for t in candidates if t not in asked]
    pool = unasked or candidates

    last_topic = state.strategy_state.recent_topics(1)
    if last_topic and len(pool) > 1:
        filtered = [t for t in pool if t != last_topic[0]]
        pool = filtered or pool

    is_early = state.strategy_state.question_count < _EARLY_QUESTION_THRESHOLD
    if is_early:
        weak_pool = [t for t in pool if t in state.weak_history_topics]
        if weak_pool:
            pool = weak_pool

    pool_sorted = sorted(
        pool,
        key=lambda t: state.coverage.topic_coverage.get(
            t, TopicCoverage(confidence=0, chunk_count=0)
        ).confidence,
        reverse=True,
    )
    top_pool = pool_sorted[:_TOP_N_POOL]

    updates["topic"] = random.choice(top_pool)
    return updates

def retrieve_evidence(state: QuestionGenState) -> dict:
    """현재 topic으로 근거를 검색한다."""
    chunks = search_evidence(query=state.topic, topic=state.topic, k=5, user_id=state.user_id)
    reliable = [c for c in chunks if c.confidence >= _EVIDENCE_CONFIDENCE_THRESHOLD]
    return {"evidence_chunks": reliable}

def generate(state: QuestionGenState) -> dict:
    """근거를 바탕으로 질문을 생성한다 (또는 재생성한다)."""
    updates: dict = {}
    if state.validation_failed:
        updates["regenerate_count"] = state.regenerate_count + 1

    context = (
        "\n".join(f"- {c.text}" for c in state.evidence_chunks)
        if state.evidence_chunks
        else "(관련 근거 없음. 근거를 인용하지 말고 일반적인 개념 질문으로 만들 것)"
    )
    asked_block = (
        "\n".join(f"- {t}" for t in state.strategy_state.asked_question_texts)
        if state.strategy_state.asked_question_texts
        else "(없음)"
    )
    retry_note = (
        f"\n\n주의: 이전 시도가 다음 이유로 거부됨 - {state.validation_reason}. "
        "이 문제를 피해서 다시 생성할 것."
        if state.validation_reason
        else ""
    )

    user_prompt = f"""\
주제: {state.topic}
난이도: {state.difficulty.value}

근거:
{context}

이미 출제한 질문 (겹치지 않게):
{asked_block}
{retry_note}
"""

    llm = get_llm(temperature=0.6)
    structured_llm = llm.with_structured_output(GeneratedQuestion)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": QUESTION_GEN_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )
        updates.update({
            "generated_text": result.text,
            "generated_category": result.category,
            "validation_failed": False,
            "validation_reason": None,
        })
    except Exception:
        updates.update({
            "generated_text": f"{state.topic}에 대해 설명해 주세요.",
            "generated_category": None,
            "validation_failed": False,
            "validation_reason": None,
        })

    return updates

def validate(state: QuestionGenState) -> dict:
    """생성된 질문을 규칙 기반으로 검증한다.

    체크 항목: 
    1. 질문 1개인지, 
    2. 물음표로 끝나는지, 
    3. 이미 한 질문과 유사한지.

    "근거 없는 단정 표현" 검증은 룰 기반으로 신뢰성 있게 판단하기 어려워
    (의미 비교가 필요함) 고민이 필요, 다루지 않는다. 
    RAG와 4단계의 reliable_chunks 필터링이 1차 방어선 역할을 한다.
    """
    text = (state.generated_text or "").strip()

    if text.count("?") != 1:
        return {"validation_failed": True, "validation_reason": "질문이 정확히 1개가 아님"}

    if not text.endswith("?"):
        return {"validation_failed": True, "validation_reason": "물음표로 끝나지 않음"}

    for asked in state.strategy_state.asked_question_texts:
        if _too_similar(text, asked):
            return {"validation_failed": True, "validation_reason": "이미 한 질문과 유사함"}

    return {"validation_failed": False, "validation_reason": None}


def _too_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """두 문장의 유사도가 threshold 이상이면 True (문자 집합 기반 근사 비교). validate 의 헬퍼 함수"""
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return False
    overlap = len(set_a & set_b) / len(set_a | set_b)
    return overlap >= threshold

def build_result(state: QuestionGenState) -> dict:
    """검증을 통과한 질문으로 최종 Question을 조립한다."""
    return {
        "result": Question(
            question_id=str(uuid4()),
            text=state.generated_text,
            topic=state.topic,
            difficulty=state.difficulty,
            kind=QuestionKind.MAIN,
            category=state.generated_category,
            evidence_ids=[c.chunk_id for c in state.evidence_chunks],
            parent_question_id=None,
        )
    }

def route_after_retrieve(state: QuestionGenState) -> str:
    """(Edge)근거 조회 후 다음으로 갈 노드를 결정한다.

    근거가 있으면 generate로 진행한다. 근거가 없으면 pick_topic으로
    되돌아가 대체 주제를 찾되, retry_count가 상한에 도달하면 더 이상
    재시도하지 않고 그대로 generate로 넘긴다.
    """
    if state.evidence_chunks:
        return "generate"
    if state.retry_count >= _MAX_RETRY:
        return "generate"
    return "pick_topic"

def route_after_validate(state: QuestionGenState) -> str:
    """(Edge)검증 후 다음으로 갈 노드를 결정한다.

    통과했으면 build_result로 간다. 실패했으면 재생성을 시도하되,
    regenerate_count가 상한에 도달했으면 더 이상 재시도하지 않고
    그대로 build_result로 간다.
    """
    if not state.validation_failed:
        return "build_result"
    if state.regenerate_count >= _MAX_REGENERATE:
        return "build_result"
    return "generate"

def get_compiled_graph():
    """질문 생성 그래프를 조립하고 컴파일해서 반환한다."""
    graph = StateGraph(QuestionGenState)

    graph.add_node("pick_topic", pick_topic)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("build_result", build_result)

    graph.set_entry_point("pick_topic")

    graph.add_edge("pick_topic", "retrieve_evidence")
    graph.add_conditional_edges(
        "retrieve_evidence",
        route_after_retrieve,
        {"generate": "generate", "pick_topic": "pick_topic"},
    )
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"build_result": "build_result", "generate": "generate"},
    )
    graph.add_edge("build_result", END)

    return graph.compile()