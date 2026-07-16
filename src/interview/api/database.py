# SQLAlchemy에서 DB 연결 엔진을 만들 때 사용하는 함수
from sqlalchemy import create_engine, text

# DB 세션을 만들기 위한 sessionmaker,
# ORM 모델들의 부모 클래스 역할을 하는 declarative_base를 가져옴
from sqlalchemy.orm import sessionmaker, declarative_base

from interview.config import settings

# 실제 DB와 연결하는 엔진 생성
# SQLAlchemy가 이 엔진을 통해 PostgreSQL과 통신함
engine = create_engine(settings.database_url)


# DB 작업을 할 때 사용할 세션 생성기
# 세션은 DB와 대화하는 작업 단위라고 보면 됨
SessionLocal = sessionmaker(
    autocommit=False,  # 자동 커밋 비활성화. 직접 commit 해야 DB에 반영됨
    autoflush=False,   # 자동 flush 비활성화. 필요할 때 명시적으로 반영
    bind=engine        # 위에서 만든 DB 엔진과 연결
)


# ORM 모델 클래스들의 부모 클래스
# 앞으로 만드는 User 같은 모델은 이 Base를 상속받아야 함
Base = declarative_base()


def create_missing_tables() -> None:
    """SQLAlchemy 모델에 정의됐지만 DB에 없는 테이블을 생성한다.

    FastAPI 애플리케이션 시작 시 호출하는 초기화 함수다. 각 모델 모듈을
    먼저 import해 모든 테이블을 Base.metadata에 등록한 다음 create_all을
    실행한다. 이미 존재하는 테이블과 데이터는 변경하지 않는다.

    Returns:
        None.
    """
    from interview.api.auth import model as auth_model
    from interview.api.evidence import model as evidence_model
    from interview.api.rubric import model as rubric_model
    from interview.api.interviews import model as interviews_model
    from interview.api.users import model as users_model

    _ = (auth_model, evidence_model, rubric_model, interviews_model, users_model)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    _migrate_legacy_rubric_schema()


def _migrate_legacy_rubric_schema() -> None:
    """기존 rubric vector 테이블을 rubric set 구조로 안전하게 보정한다.

    ``create_all``은 이미 존재하는 테이블에 새 컬럼을 추가하지 않는다. 초기
    rubric 구현으로 생성된 ``rubric_vector_records``에는 ``rubric_set_id``가
    없으므로, 기존 레코드에 대응하는 set을 만든 뒤 외래 키를 채운다. 모든
    SQL은 재실행 가능하게 작성해 애플리케이션 시작 때마다 호출해도 안전하다.
    """
    with engine.begin() as connection:
        table_exists = connection.scalar(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = current_schema()
                      AND table_name = 'rubric_vector_records'
                )
                """
            )
        )
        if not table_exists:
            return

        connection.execute(
            text(
                """
                ALTER TABLE rubric_vector_records
                ADD COLUMN IF NOT EXISTS rubric_set_id INTEGER
                """
            )
        )
        connection.execute(
            text(
                f"""
                ALTER TABLE rubric_sets
                ADD COLUMN IF NOT EXISTS question_embedding
                    vector({settings.embedding_dimensions})
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO rubric_sets (
                    question_id,
                    topic,
                    question,
                    rubric_version,
                    status,
                    created_at,
                    updated_at
                )
                SELECT DISTINCT ON (question_id, rubric_version)
                    question_id,
                    topic,
                    question,
                    rubric_version,
                    'pending',
                    created_at,
                    created_at
                FROM rubric_vector_records
                WHERE rubric_set_id IS NULL
                ORDER BY question_id, rubric_version, created_at
                ON CONFLICT (question_id, rubric_version) DO NOTHING
                """
            )
        )
        connection.execute(
            text(
                """
                UPDATE rubric_vector_records AS record
                SET rubric_set_id = rubric_set.id
                FROM rubric_sets AS rubric_set
                WHERE record.rubric_set_id IS NULL
                  AND rubric_set.question_id = record.question_id
                  AND rubric_set.rubric_version = record.rubric_version
                """
            )
        )
        connection.execute(
            text(
                """
                UPDATE rubric_sets AS rubric_set
                SET status = 'verified', updated_at = CURRENT_TIMESTAMP
                WHERE rubric_set.status = 'pending'
                  AND EXISTS (
                      SELECT 1
                      FROM rubric_vector_records AS record
                      WHERE record.rubric_set_id = rubric_set.id
                  )
                """
            )
        )
        connection.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM rubric_vector_records
                        WHERE rubric_set_id IS NULL
                    ) THEN
                        ALTER TABLE rubric_vector_records
                        ALTER COLUMN rubric_set_id SET NOT NULL;
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conname = 'fk_rubric_vector_records_set'
                    ) THEN
                        ALTER TABLE rubric_vector_records
                        ADD CONSTRAINT fk_rubric_vector_records_set
                        FOREIGN KEY (rubric_set_id)
                        REFERENCES rubric_sets(id)
                        ON DELETE CASCADE;
                    END IF;
                END
                $$
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                    ix_rubric_vector_records_rubric_set_id
                ON rubric_vector_records (rubric_set_id)
                """
            )
        )

# FastAPI에서 DB 세션을 의존성 주입으로 사용하기 위한 함수
# API 요청이 들어올 때 DB 세션을 하나 만들고,
# 요청 처리가 끝나면 세션을 닫아줌
def get_db():
    # DB 세션 생성
    db = SessionLocal()

    try:
        # router 함수에 db 세션을 전달
        yield db

    finally:
        # 요청 처리가 끝나면 DB 세션 닫기
        db.close()
