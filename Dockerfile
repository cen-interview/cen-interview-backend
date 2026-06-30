# 1단계: uv 환경 빌더 stage
FROM ghcr.io/astral-sh/uv:python3.11-alpine AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# 의존성만 먼저 설치 (소스 변경과 무관한 캐시 레이어)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 소스 복사 후 프로젝트 자체를 설치한다.
# (--no-install-project 만 쓰면 의존성만 깔리고 interview 패키지는 venv에
#  안 들어가서 런타임에 ModuleNotFoundError: No module named 'interview' 가 난다)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# 2단계: 실행용 stage (경량화)
FROM python:3.11-alpine
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/

# 환경변수 설정 (uv 가상환경 활성화 및 호스트 바인딩)
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "interview.api.main:app", "--host", "0.0.0.0", "--port", "8000"]