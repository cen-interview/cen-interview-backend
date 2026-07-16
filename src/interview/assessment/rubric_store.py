"""공개 rubric의 저장과 답변-기준 유사도 검색."""

from collections import OrderedDict
from collections.abc import Callable
from hashlib import sha1
import re
from threading import RLock
from time import monotonic
from typing import Any

from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from interview.api.database import SessionLocal
from interview.api.rubric.model import (
    RubricQuestionAlias,
    RubricSetRecord,
    RubricVectorRecord,
)
from interview.config import settings
from interview.schemas.rubric import (
    RubricCandidate,
    RubricMatchResult,
    RubricSource,
)


QUESTION_DUPLICATE_THRESHOLD = 0.9
QUESTION_SEARCH_TOP_K = 5
CRITERION_DUPLICATE_THRESHOLD = 0.9
QUESTION_CACHE_TTL_SECONDS = 900.0
QUESTION_CACHE_MAX_SIZE = 512


class RubricStore:
    """공개 rubric을 질문별로 검색한다.

    memory backend는 테스트용이고, 운영 기본값은 pgvector다. 저장은 최종
    공유 동의 이후에만 호출해야 하며, 조회는 사용자 ID 없이 공용 namespace를
    사용한다.
    """

    def __init__(
        self,
        database_url: str | None = None,
        backend: str | None = None,
        embedding_client: Any | None = None,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        self.backend = (backend or settings.evidence_store_backend).lower()
        self.embedding_client = embedding_client
        self._candidates: list[tuple[RubricCandidate, list[float]]] = []
        self._cache_lock = RLock()
        self._index_version = 0
        self._question_embedding_cache: OrderedDict[
            str, tuple[float, list[float]]
        ] = OrderedDict()
        self._question_candidate_cache: OrderedDict[
            tuple[int, str, str], tuple[float, list[tuple[int, float]]]
        ] = OrderedDict()
        if session_factory is not None:
            self._session_factory = session_factory
        elif database_url is not None and database_url != settings.database_url:
            engine = create_engine(database_url)
            self._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        else:
            self._session_factory = SessionLocal

    def add_candidate(self, candidate: RubricCandidate) -> None:
        """공개 동의가 끝난 rubric 후보를 기준별로 저장한다."""
        texts = [candidate.question, *[c.description for c in candidate.criteria]]
        embeddings = self._embed_documents(texts)
        question_embedding, criterion_embeddings = embeddings[0], embeddings[1:]
        if self.backend == "memory":
            self._candidates = [item for item in self._candidates if item[0].question_id != candidate.question_id]
            self._candidates.extend(
                (RubricCandidate(
                    question_id=candidate.question_id,
                    topic=candidate.topic,
                    question=candidate.question,
                    criteria=[criterion],
                    rubric_version=candidate.rubric_version,
                ), embedding)
                for criterion, embedding in zip(
                    candidate.criteria, criterion_embeddings, strict=True
                )
            )
            self._invalidate_candidate_cache()
            return

        with self._session_factory() as db:
            rubric_set = db.scalar(select(RubricSetRecord).where(
                RubricSetRecord.question_id == candidate.question_id,
                RubricSetRecord.rubric_version == candidate.rubric_version,
            ))
            if rubric_set is None:
                similar_sets = self._find_question_candidates(
                    db,
                    question_embedding=question_embedding,
                    topic=candidate.topic,
                    top_k=1,
                )
                if (
                    similar_sets
                    and similar_sets[0][1] >= QUESTION_DUPLICATE_THRESHOLD
                ):
                    rubric_set = similar_sets[0][0]
            if rubric_set is None:
                rubric_set = RubricSetRecord(
                    question_id=candidate.question_id,
                    topic=candidate.topic,
                    question=candidate.question,
                    question_embedding=question_embedding,
                    rubric_version=candidate.rubric_version,
                    status="verified",
                )
                db.add(rubric_set)
                db.flush()
            else:
                rubric_set.status = "verified"
                alias = db.scalar(
                    select(RubricQuestionAlias).where(
                        RubricQuestionAlias.rubric_set_id == rubric_set.id,
                        RubricQuestionAlias.source_question_id
                        == candidate.question_id,
                    )
                )
                if (
                    candidate.question_id != rubric_set.question_id
                    and alias is None
                ):
                    db.add(
                        RubricQuestionAlias(
                            rubric_set_id=rubric_set.id,
                            source_question_id=candidate.question_id,
                            question_text=candidate.question,
                            question_embedding=question_embedding,
                        )
                    )

            existing_records = db.scalars(
                select(RubricVectorRecord).where(
                    RubricVectorRecord.rubric_set_id == rubric_set.id
                )
            ).all()
            candidate_required_records: list[RubricVectorRecord] = []
            for criterion, embedding in zip(
                candidate.criteria, criterion_embeddings, strict=True
            ):
                record = max(
                    existing_records,
                    key=lambda item: _cosine_similarity(
                        embedding, list(item.embedding)
                    ),
                    default=None,
                )
                if (
                    record is not None
                    and _cosine_similarity(embedding, list(record.embedding))
                    < CRITERION_DUPLICATE_THRESHOLD
                ):
                    record = None
                criterion_id = (
                    record.criterion_id
                    if record is not None
                    else _semantic_criterion_id(criterion.description)
                )
                values = {
                    "rubric_set_id": rubric_set.id,
                    "question_id": rubric_set.question_id,
                    "topic": rubric_set.topic,
                    "question": rubric_set.question,
                    "criterion_id": criterion_id,
                    "criterion_text": criterion.description,
                    "required": criterion.required,
                    "weight": criterion.weight,
                    "rubric_version": candidate.rubric_version,
                    "embedding": embedding,
                }
                if record is None:
                    record = RubricVectorRecord(**values)
                    db.add(record)
                    existing_records.append(record)
                else:
                    values["weight"] = max(record.weight, criterion.weight)
                    for key, value in values.items():
                        setattr(record, key, value)
                if criterion.required:
                    candidate_required_records.append(record)

            # 과거의 required 플래그를 OR로 누적하지 않고,
            # 검증된 최신 후보의 1~2개 필수 기준을 반영한다.
            for record in existing_records:
                record.required = any(
                    record is required_record
                    for required_record in candidate_required_records
                )
            db.commit()
        self._invalidate_candidate_cache()

    def prefetch_question(
        self,
        question_id: str,
        question_text: str,
        topic: str,
    ) -> None:
        """질문 노출 시 임베딩과 Top-K 후보를 미리 캐시에 적재한다.

        prefetch 실패는 실제 답변 평가에서 다시 조회할 수 있으므로 호출자가
        면접 흐름을 중단하지 않도록 예외를 처리해야 한다.
        """
        if self.backend == "memory":
            self._get_question_embedding(question_text)
            return

        with self._session_factory() as db:
            exact = db.scalar(
                select(RubricSetRecord.id).where(
                    RubricSetRecord.question_id == question_id,
                    RubricSetRecord.status == "verified",
                )
            )
            if exact is not None:
                return
            question_embedding = self._get_question_embedding(question_text)
            self._get_question_candidates_cached(
                db,
                question_embedding=question_embedding,
                question_text=question_text,
                topic=topic,
            )

    def filter_novel_questions(
        self,
        sources: list[RubricSource],
        *,
        threshold: float = QUESTION_DUPLICATE_THRESHOLD,
    ) -> list[RubricSource]:
        """Return only questions not already represented in the same topic.

        Source questions are embedded in one batch. Existing DB vectors and
        already-selected questions from the current interview are then used to
        remove semantic duplicates before any rubric-generation LLM is called.
        """
        if not sources:
            return []

        source_embeddings = self._embed_documents([
            source.question for source in sources
        ])
        selected: list[tuple[RubricSource, list[float]]] = []

        if self.backend == "memory":
            existing_questions: dict[tuple[str, str], RubricCandidate] = {}
            for candidate, _ in self._candidates:
                key = (
                    _normalize_topic(candidate.topic),
                    candidate.question_id,
                )
                existing_questions[key] = candidate
            existing_rows = list(existing_questions.values())
            existing_embeddings = self._embed_documents([
                candidate.question for candidate in existing_rows
            ]) if existing_rows else []

            for source, embedding in zip(
                sources,
                source_embeddings,
                strict=True,
            ):
                topic_key = _normalize_topic(source.topic)
                duplicate_in_store = any(
                    _normalize_topic(candidate.topic) == topic_key
                    and _cosine_similarity(embedding, candidate_embedding)
                    >= threshold
                    for candidate, candidate_embedding in zip(
                        existing_rows,
                        existing_embeddings,
                        strict=True,
                    )
                )
                if duplicate_in_store or _duplicates_selected_source(
                    source,
                    embedding,
                    selected,
                    threshold=threshold,
                ):
                    continue
                selected.append((source, embedding))
            return [source for source, _ in selected]

        with self._session_factory() as db:
            for source, embedding in zip(
                sources,
                source_embeddings,
                strict=True,
            ):
                existing = self._find_question_candidates(
                    db,
                    question_embedding=embedding,
                    topic=source.topic,
                    top_k=1,
                )
                if existing and existing[0][1] >= threshold:
                    continue
                if _duplicates_selected_source(
                    source,
                    embedding,
                    selected,
                    threshold=threshold,
                ):
                    continue
                selected.append((source, embedding))

        return [source for source, _ in selected]

    def match(
        self,
        question_id: str,
        answer_text: str,
        *,
        question_text: str | None = None,
        topic: str | None = None,
        question_threshold: float = 0.8,
        threshold: float = 0.8,
    ) -> RubricMatchResult | None:
        """유사한 질문의 공개 rubric을 찾고 필수 기준과 답변을 비교한다."""
        if self.backend == "memory":
            rows = [item for item in self._candidates if item[0].question_id == question_id]
            question_similarity = 1.0
            if not rows and question_text and topic:
                candidates_by_question: dict[str, RubricCandidate] = {}
                for candidate, _ in self._candidates:
                    if candidate.topic == topic:
                        candidates_by_question[candidate.question_id] = candidate
                if candidates_by_question:
                    query_embedding = self._embed_documents([question_text])[0]
                    scored = [
                        (
                            _cosine_similarity(
                                query_embedding,
                                self._embed_documents([candidate.question])[0],
                            ),
                            candidate.question_id,
                        )
                        for candidate in candidates_by_question.values()
                    ]
                    question_similarity, matched_id = max(scored)
                    if question_similarity >= question_threshold:
                        rows = [
                            item
                            for item in self._candidates
                            if item[0].question_id == matched_id
                        ]
            if not rows:
                return None
            answer_embeddings = self._embed_documents(
                _answer_segments(answer_text)
            )
            similarities = {
                candidate.criteria[0].criterion_id: max(
                    _cosine_similarity(answer_embedding, embedding)
                    for answer_embedding in answer_embeddings
                )
                for candidate, embedding in rows
            }
            required = [candidate.criteria[0] for candidate, _ in rows if candidate.criteria[0].required]
            return RubricMatchResult(
                question_id=question_id,
                rubric_version=rows[0][0].rubric_version,
                criterion_similarities=similarities,
                required_criteria_count=len(required),
                matched_required_count=sum(similarities[c.criterion_id] >= threshold for c in required),
                threshold=threshold,
                matched_rubric_question_id=rows[0][0].question_id,
                question_similarity=question_similarity,
            )

        answer_segments = _answer_segments(answer_text)
        answer_embeddings: list[list[float]] | None = None
        with self._session_factory() as db:
            exact_rubric_set = db.scalar(
                select(RubricSetRecord).where(
                    RubricSetRecord.question_id == question_id,
                    RubricSetRecord.status == "verified",
                )
            )
            candidates: list[tuple[RubricSetRecord, float]] = []
            if exact_rubric_set is not None:
                candidates = [(exact_rubric_set, 1.0)]
            elif question_text and topic:
                question_embedding = self._question_embedding_from_cache(
                    question_text
                )
                if question_embedding is None:
                    embeddings = self._embed_documents(
                        [question_text, *answer_segments]
                    )
                    question_embedding = embeddings[0]
                    answer_embeddings = embeddings[1:]
                    self._cache_question_embedding(
                        question_text, question_embedding
                    )
                candidates = [
                    item
                    for item in self._get_question_candidates_cached(
                        db,
                        question_embedding=question_embedding,
                        question_text=question_text,
                        topic=topic,
                    )
                    if item[1] >= question_threshold
                ]
            if not candidates:
                return None
            if answer_embeddings is None:
                answer_embeddings = self._embed_documents(answer_segments)
            matches = [
                self._match_candidate(
                    db,
                    question_id=question_id,
                    rubric_set=rubric_set,
                    question_similarity=question_similarity,
                    answer_embeddings=answer_embeddings,
                    threshold=threshold,
                )
                for rubric_set, question_similarity in candidates
            ]
        valid_matches = [match for match in matches if match is not None]
        if not valid_matches:
            return None
        return max(
            valid_matches,
            key=lambda match: (
                match.is_sufficient,
                match.matched_required_count,
                match.required_coverage,
                match.question_similarity or 0.0,
            ),
        )

    def _get_question_candidates_cached(
        self,
        db: Session,
        *,
        question_embedding: list[float],
        question_text: str,
        topic: str,
    ) -> list[tuple[RubricSetRecord, float]]:
        cache_key = (
            self._index_version,
            _normalize_topic(topic),
            _normalize_question(question_text),
        )
        now = monotonic()
        with self._cache_lock:
            cached = self._question_candidate_cache.get(cache_key)
            if cached is not None and cached[0] > now:
                self._question_candidate_cache.move_to_end(cache_key)
                cached_rows = cached[1]
            else:
                if cached is not None:
                    del self._question_candidate_cache[cache_key]
                cached_rows = None
        if cached_rows is None:
            found = self._find_question_candidates(
                db,
                question_embedding=question_embedding,
                topic=topic,
                top_k=QUESTION_SEARCH_TOP_K,
            )
            cached_rows = [(rubric_set.id, score) for rubric_set, score in found]
            with self._cache_lock:
                self._question_candidate_cache[cache_key] = (
                    now + QUESTION_CACHE_TTL_SECONDS,
                    cached_rows,
                )
                self._trim_cache(self._question_candidate_cache)

        if not cached_rows:
            return []
        ids = [rubric_set_id for rubric_set_id, _ in cached_rows]
        rubric_sets = db.scalars(
            select(RubricSetRecord).where(RubricSetRecord.id.in_(ids))
        ).all()
        by_id = {rubric_set.id: rubric_set for rubric_set in rubric_sets}
        return [
            (by_id[rubric_set_id], score)
            for rubric_set_id, score in cached_rows
            if rubric_set_id in by_id
        ]

    def _get_question_embedding(self, question_text: str) -> list[float]:
        cached = self._question_embedding_from_cache(question_text)
        if cached is not None:
            return cached
        embedding = self._embed_documents([question_text])[0]
        self._cache_question_embedding(question_text, embedding)
        return embedding

    def _question_embedding_from_cache(
        self, question_text: str
    ) -> list[float] | None:
        key = _normalize_question(question_text)
        now = monotonic()
        with self._cache_lock:
            cached = self._question_embedding_cache.get(key)
            if cached is None:
                return None
            if cached[0] <= now:
                del self._question_embedding_cache[key]
                return None
            self._question_embedding_cache.move_to_end(key)
            return cached[1]

    def _cache_question_embedding(
        self, question_text: str, embedding: list[float]
    ) -> None:
        key = _normalize_question(question_text)
        with self._cache_lock:
            self._question_embedding_cache[key] = (
                monotonic() + QUESTION_CACHE_TTL_SECONDS,
                embedding,
            )
            self._trim_cache(self._question_embedding_cache)

    def _invalidate_candidate_cache(self) -> None:
        with self._cache_lock:
            self._index_version += 1
            self._question_candidate_cache.clear()

    @staticmethod
    def _trim_cache(cache: OrderedDict[Any, Any]) -> None:
        while len(cache) > QUESTION_CACHE_MAX_SIZE:
            cache.popitem(last=False)

    def _find_question_candidates(
        self,
        db: Session,
        *,
        question_embedding: list[float],
        topic: str,
        top_k: int,
    ) -> list[tuple[RubricSetRecord, float]]:
        canonical_rows = db.execute(
            select(
                RubricSetRecord,
                RubricSetRecord.question_embedding.cosine_distance(
                    question_embedding
                ).label("distance"),
            )
            .where(
                RubricSetRecord.status == "verified",
                RubricSetRecord.topic == topic,
                RubricSetRecord.question_embedding.is_not(None),
            )
            .order_by("distance")
            .limit(top_k)
        ).all()
        alias_rows = db.execute(
            select(
                RubricSetRecord,
                RubricQuestionAlias.question_embedding.cosine_distance(
                    question_embedding
                ).label("distance"),
            )
            .join(
                RubricQuestionAlias,
                RubricQuestionAlias.rubric_set_id == RubricSetRecord.id,
            )
            .where(
                RubricSetRecord.status == "verified",
                RubricSetRecord.topic == topic,
            )
            .order_by("distance")
            .limit(top_k)
        ).all()
        best_by_set: dict[int, tuple[RubricSetRecord, float]] = {}
        for rubric_set, distance in [*canonical_rows, *alias_rows]:
            similarity = 1.0 - float(distance)
            current = best_by_set.get(rubric_set.id)
            if current is None or similarity > current[1]:
                best_by_set[rubric_set.id] = (rubric_set, similarity)
        return sorted(
            best_by_set.values(), key=lambda item: item[1], reverse=True
        )[:top_k]

    def _match_candidate(
        self,
        db: Session,
        *,
        question_id: str,
        rubric_set: RubricSetRecord,
        question_similarity: float,
        answer_embeddings: list[list[float]],
        threshold: float,
    ) -> RubricMatchResult | None:
        records = db.scalars(
            select(RubricVectorRecord).where(
                RubricVectorRecord.rubric_set_id == rubric_set.id
            )
        ).all()
        if not records:
            return None
        similarities = {
            record.criterion_id: max(
                _cosine_similarity(answer_embedding, list(record.embedding))
                for answer_embedding in answer_embeddings
            )
            for record in records
        }
        required = [record for record in records if record.required]
        return RubricMatchResult(
            question_id=question_id,
            rubric_version=records[0].rubric_version,
            criterion_similarities=similarities,
            required_criteria_count=len(required),
            matched_required_count=sum(
                similarities[record.criterion_id] >= threshold
                for record in required
            ),
            threshold=threshold,
            matched_rubric_question_id=rubric_set.question_id,
            question_similarity=question_similarity,
        )

    def backfill_question_embeddings(self) -> int:
        """기존 verified rubric set의 누락된 질문 임베딩을 채운다."""
        if self.backend == "memory":
            return 0

        with self._session_factory() as db:
            rubric_sets = db.scalars(
                select(RubricSetRecord).where(
                    RubricSetRecord.status == "verified",
                    RubricSetRecord.question_embedding.is_(None),
                )
            ).all()
            if not rubric_sets:
                return 0
            embeddings = self._embed_documents(
                [rubric_set.question for rubric_set in rubric_sets]
            )
            for rubric_set, embedding in zip(
                rubric_sets, embeddings, strict=True
            ):
                rubric_set.question_embedding = embedding
            db.commit()
            updated_count = len(rubric_sets)
        self._invalidate_candidate_cache()
        return updated_count

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_embeddings().embed_documents(texts)

    def _get_embeddings(self) -> Any:
        if self.embedding_client is None:
            self.embedding_client = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
                dimensions=settings.embedding_dimensions,
            )
        return self.embedding_client


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    import math
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def _normalize_topic(topic: str) -> str:
    """Normalize topic spelling for comparisons within one interview."""
    return " ".join(topic.strip().casefold().split())


def _normalize_question(question: str) -> str:
    """표현 캐시 키가 공백과 대소문자 차이로 분리되지 않게 정규화한다."""
    return " ".join(question.strip().casefold().split())


def _duplicates_selected_source(
    source: RubricSource,
    embedding: list[float],
    selected: list[tuple[RubricSource, list[float]]],
    *,
    threshold: float,
) -> bool:
    topic_key = _normalize_topic(source.topic)
    return any(
        _normalize_topic(selected_source.topic) == topic_key
        and _cosine_similarity(embedding, selected_embedding) >= threshold
        for selected_source, selected_embedding in selected
    )


def _answer_segments(answer_text: str) -> list[str]:
    """긴 답변의 의미가 희석되지 않도록 전체 답변과 문장 조각을 반환한다."""
    normalized = answer_text.strip()
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|\n+", normalized)
        if segment.strip()
    ]
    return list(dict.fromkeys([normalized, *segments])) if normalized else [""]


def _semantic_criterion_id(description: str) -> str:
    digest = sha1(description.strip().lower().encode("utf-8")).hexdigest()
    return f"criterion-{digest[:16]}"


def _cosine_distance(left: list[float], right: list[float]) -> float:
    return 1.0 - _cosine_similarity(left, right)


_rubric_store: RubricStore | None = None


def get_rubric_store() -> RubricStore:
    global _rubric_store
    if _rubric_store is None:
        _rubric_store = RubricStore()
    return _rubric_store
