"""면접 최종 결과를 저장하는 SQLAlchemy 모델."""
from enum import Enum

# DB 컬럼을 정의할 때 사용하는 타입
from sqlalchemy import (
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Integer,
)

# PostgreSQL의 JSONB 컬럼 타입
from sqlalchemy.dialects.postgresql import JSONB

# DB 서버 기준 현재 시간을 사용하기 위한 함수
from sqlalchemy.sql import func

# 모든 SQLAlchemy 모델이 상속하는 Base
from interview.api.database import Base
from interview.schemas.events import Mode

class InterviewSessionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# 면접 시작과 끝 
class InterviewSession(Base):
    __tablename__ = "interview_sessions"
# Integer + primary_key=True이면 기본적으로 auto increment가 자동 적용
    session_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    mode = Column(
        SQLEnum(
            Mode,
            name="interview_mode",
            values_callable=lambda enum: [
                mode.value for mode in enum
            ],
        ),
        nullable=False,
    )

    status = Column(
        SQLEnum(
            InterviewSessionStatus,
            name="interview_session_status",
            values_callable=lambda enum: [
                status.value for status in enum
            ],
        ),
        nullable=False,
        default=InterviewSessionStatus.IN_PROGRESS,
        server_default=InterviewSessionStatus.IN_PROGRESS.value,
    )

    started_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    ended_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )


# 면접 최종 결과 테이블 모델
# 면접 한 세션이 종료될 때 최종 리포트 하나가 저장된다.
class InterviewResult(Base):

    # 실제 DB에 생성될 테이블 이름
    __tablename__ = "interview_results"

    # 면접 결과 고유 ID
    # primary_key=True: 기본키
    # index=True: 결과 ID 조회를 위한 인덱스
    # Integer + primary_key=True이면 기본적으로 auto increment가 자동 적용
    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )


    # 면접 세션 ID
    # 같은 세션의 결과가 중복 저장되지 않도록 unique로 설정한다.
    session_id = Column(
        Integer,
        ForeignKey("interview_sessions.session_id"),
        nullable=False,
        unique=True,
        index=True,
    )

    # 면접 최종 종합 점수
    # 결과 목록에서 JSON을 열지 않고 점수를 조회할 수 있도록
    # 별도 컬럼으로 저장한다.
    overall_score = Column(
        Float,
        nullable=False,
    )

    # FinalReport 전체 데이터
    # summary, strengths, improvement_points,
    # learning_recommendations, evaluations 등을 JSON으로 저장한다.
    final_report_json = Column(
        JSONB,
        nullable=False,
    )

    # 주제별 점수
    # 다음 면접에서 약점 주제를 조회하기 위해 별도 저장한다.
    # 예: {"JPA": 72.5, "Redis": 61.0}
    topic_scores_json = Column(
        JSONB,
        nullable=False,
    )

    # 면접 결과 저장 시간
    # DB 서버 시간을 기준으로 자동 생성한다.
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
