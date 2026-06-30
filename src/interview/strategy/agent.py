"""Strategy Agent.

면접 질문의 방향·순서·난이도를 계속 조정하는 전략 담당. LangGraph 노드에서
호출되는 진입점들을 모아둔다. 실제 생성/난이도 로직은 question_gen / difficulty 에.
"""

from interview.schemas.evidence import CoverageMap
from interview.schemas.question import Question
from interview.schemas.signals import AnswerQualitySignal
from interview.strategy import difficulty, question_gen  # noqa: F401 (TODO 담당 B: question_gen 연결 시 사용)
from interview.strategy.state import StrategyState


class StrategyAgent:
    def __init__(self, coverage: CoverageMap) -> None:
        self.state = StrategyState()
        self.coverage = coverage or CoverageMap(topic_confidence={}, updated_at=None)

        # [Stub 전용] 시뮬레이션용 가짜 질문 풀
        self._dummy_topics = ["Java Memory", "Spring Boot", "Database"]
        self._dummy_questions = {
            "Java Memory": "Java의 JVM 메모리 구조 중 Stack과 Heap의 차이에 대해 설명해주세요.",
            "Spring Boot": "Spring Boot에서 @RestController와 @Controller의 차이는 무엇인가요?",
            "Database": "MySQL에서 Index를 사용하는 이유와 단점에 대해 설명해주세요."
        }
        self._topic_idx = 0

    def next_question(self, last_signal: AnswerQualitySignal | None) -> Question:
        """다음 일반 질문 선택."""
        diff = difficulty.next_difficulty(self.state, last_signal)
        topic = self._pick_topic()
        
        # [실전 전환용 주석] question_gen이 완성되면 아래 주석을 풀고 Stub 리턴을 주석 처리하세요.
        # return question_gen.generate_question(topic, diff)
        
        # [현재 Stub 작동]
        text = self._dummy_questions.get(topic, "공통 질문입니다.")
        return Question(
            question_id=f"q_main_{self._topic_idx}",
            text=text,
            topic=topic,
            difficulty=diff,
            kind="main"
        )

    def next_follow_up(self, topic: str, missing_keywords: list[str]) -> Question:
        """Interviewer 요청에 따라 꼬리 질문 생성."""
        # [실전 전환용 주석]
        # return question_gen.generate_follow_up(topic, missing_keywords)
        
        # [현재 Stub 작동]
        keywords_str = ", ".join(missing_keywords) if missing_keywords else "상세한 내용"
        return Question(
            question_id=f"q_follow_{self._topic_idx}",
            text=f"[{topic} 꼬리질문] 방금 답변해주신 내용 중에서 {keywords_str} 부분에 대해 조금만 더 깊게 설명해주실 수 있나요?",
            topic=topic,
            difficulty="medium",
            kind="follow_up"
        )

    def next_hint(self, topic: str) -> Question:
        """막힘 상황 힌트성 질문 생성."""
        # [실전 전환용 주석]
        # return question_gen.generate_hint(topic)
        
        # [현재 Stub 작동]
        return Question(
            question_id=f"q_hint_{self._topic_idx}",
            text=f"[{topic} 힌트] 이 개념이 실무에서 왜 필요하게 되었는지 목적을 떠올려보시면 도움이 될 것 같습니다.",
            topic=topic,
            difficulty="easy",
            kind="hint"
        )

    def _pick_topic(self) -> str:
        """다음 주제 선택 (주제 쏠림 방지)."""
        # TODO(담당 B): 원래 구현되어야 할 주석 로직 유지
        # coverage + 이미 물어본 주제 분포로 다음 주제 결정
        
        # [현재 Stub 작동] 에러 안 나게 순환하며 토픽 뱉기
        topic = self._dummy_topics[self._topic_idx % len(self._dummy_topics)]
        self._topic_idx += 1
        return topic