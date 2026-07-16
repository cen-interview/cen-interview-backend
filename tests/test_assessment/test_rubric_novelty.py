from interview.assessment.rubric_store import RubricStore
from interview.schemas.rubric import (
    RubricCandidate,
    RubricCriterion,
    RubricSource,
)


class FakeEmbeddings:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


def _candidate(question: str = "FastAPI 비동기 함수는 언제 사용하나요?"):
    return RubricCandidate(
        question_id="stored-question",
        topic="FastAPI",
        question=question,
        criteria=[
            RubricCriterion(
                criterion_id="async-io",
                description="비동기 I/O를 설명한다.",
            )
        ],
    )


def test_novel_questions_use_topic_and_question_similarity_before_llm():
    existing = "FastAPI 비동기 함수는 언제 사용하나요?"
    duplicate = "FastAPI에서 async def는 언제 사용하나요?"
    novel = "FastAPI Depends는 어떻게 동작하나요?"
    other_topic = "Vue에서 async def는 언제 사용하나요?"
    embeddings = FakeEmbeddings({
        existing: [1.0, 0.0],
        duplicate: [0.99, 0.01],
        novel: [0.0, 1.0],
        other_topic: [1.0, 0.0],
    })
    store = RubricStore(backend="memory", embedding_client=embeddings)
    store._candidates = [(_candidate(existing), [0.0, 1.0])]
    sources = [
        RubricSource(
            question_id="duplicate",
            topic="FastAPI",
            question=duplicate,
            answer="answer",
        ),
        RubricSource(
            question_id="novel",
            topic="FastAPI",
            question=novel,
            answer="answer",
        ),
        RubricSource(
            question_id="other-topic",
            topic="Vue",
            question=other_topic,
            answer="answer",
        ),
    ]

    result = store.filter_novel_questions(sources)

    assert [source.question_id for source in result] == ["novel", "other-topic"]
    assert embeddings.calls[0] == [duplicate, novel, other_topic]


def test_current_interview_duplicate_questions_keep_first_source():
    first = "Vue Router 중첩 라우트는 어떻게 구성하나요?"
    duplicate = "Vue Router에서 중첩 경로를 구성하는 방법은?"
    embeddings = FakeEmbeddings({
        first: [1.0, 0.0],
        duplicate: [0.99, 0.01],
    })
    store = RubricStore(backend="memory", embedding_client=embeddings)
    sources = [
        RubricSource(
            question_id="first",
            topic="Vue Router",
            question=first,
            answer="answer one",
        ),
        RubricSource(
            question_id="duplicate",
            topic="vue router",
            question=duplicate,
            answer="answer two",
        ),
    ]

    result = store.filter_novel_questions(sources)

    assert [source.question_id for source in result] == ["first"]
