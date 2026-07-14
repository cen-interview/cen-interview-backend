"""면접 세션과 최종 결과를 저장하는 SQLAlchemy ORM 모델.

이 모듈은 면접의 시작부터 종료까지의 상태를 관리하는
InterviewSession 모델과, 면접 종료 후 생성된 최종 리포트를 저장하는
InterviewResult 모델을 정의한다.

주요 구성:
    InterviewSessionStatus:
        면접 세션의 진행 상태를 나타내는 Enum이다.
        in_progress, completed, cancelled 상태를 제공한다.

    InterviewSession:
        사용자가 시작한 면접 세션 정보를 저장한다.
        면접 모드, 진행 상태, 시작·종료 시간과 사용자를 관리한다.

        session_id는 DB 내부 관계에 사용하는 자동 증가 기본키이고,
        runtime_session_id는 LangGraph와 API에서 사용하는 세션 ID다.
        예를 들어 runtime_session_id에는 "sess_a12b34cd"가 저장된다.

    InterviewResult:
        면접 세션이 종료된 후 생성된 최종 평가 결과를 저장한다.
        한 면접 세션에는 하나의 결과만 저장되도록 session_id에
        unique 제약조건을 적용한다.

        최종 종합 점수는 overall_score에 별도로 저장하고,
        FinalReport 전체 내용은 final_report_json에 JSONB로 저장한다.
        다음 면접 개인화에 사용할 주제별 점수는
        topic_scores_json에 별도로 저장한다.

데이터 관계:
    users 1:N interview_sessions
    interview_sessions 1:1 interview_results

처리 흐름:
    1. 면접을 시작하면 InterviewSession을 in_progress 상태로 생성한다.
    2. runtime_session_id를 이용해 실행 중인 LangGraph 세션과 연결한다.
    3. 면접이 종료되면 InterviewResult에 최종 리포트를 저장한다.
    4. 저장 완료 후 InterviewSession 상태를 completed로 변경하고
       ended_at을 기록한다.
    5. topic_scores_json의 낮은 점수 주제는 다음 면접의 질문
       개인화에 활용할 수 있다.
"""
from enum import Enum

# DB 컬럼을 정의할 때 사용하는 타입
from sqlalchemy import (
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Integer,
    String,
)

# PostgreSQL의 JSONB 컬럼 타입
from sqlalchemy.dialects.postgresql import JSONB

# DB 서버 기준 현재 시간을 사용하기 위한 함수
from sqlalchemy.sql import func

# 모든 SQLAlchemy 모델이 상속하는 Base
from interview.api.database import Base
from interview.schemas.events import Mode

# 면접 세션의 진행 상태
class InterviewSessionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# 면접 실행 정보와 진행 상태를 저장하는 DB 모델.
class InterviewSession(Base):
    __tablename__ = "interview_sessions"
# Integer + primary_key=True이면 기본적으로 auto increment가 자동 적용
    session_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    runtime_session_id = Column(
    String(50),
        nullable=False,
        unique=True,
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


# 종료된 면접의 최종 리포트와 주제별 점수를 저장하는 DB 모델.
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
