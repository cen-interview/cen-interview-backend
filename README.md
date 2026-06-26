# Interview Agent

Notion 학습 기록 + GitHub 프로젝트를 근거로, 음성/채팅으로 진행되는 AI 기술 면접 서비스.

## 핵심 설계 원칙 (먼저 읽기)

이 프로젝트는 4명이 에이전트 1개씩 병렬로 짠다. 망하지 않는 유일한 방법은
**"계약(schemas)부터 합의하고, 남의 에이전트는 가짜(mock)로 끼워 독립 개발"** 이다.

- `src/interview/schemas/` 가 모두의 **계약**이다. 이걸 바꾸면 남이 깨지니, 변경은 **반드시 팀 합의 후 같이** 수정한다.
- 각자 자기 폴더 안에서만 작업한다 → Git 충돌 최소화.
- 공통으로 건드리는 곳은 `schemas/` 와 `interviewer/graph.py`(접착제) 둘뿐. 여기만 조심.

## 역할 분담

| 담당 | 패키지 | 책임 |
|------|--------|------|
| A | `evidence/` | Notion/GitHub 인덱싱 파이프라인 + Retrieval Tool |
| B | `strategy/` | 질문 주제/순서/난이도 결정 |
| C | `interviewer/` + `api/` | 흐름 조율, 음성/채팅 모드 흡수, **그래프 배선·세션(접착제)** |
| D | `assessment/` | 근거 기반 답변 평가 + 최종 리포트 |

> ⚠️ 오케스트레이션(`graph.py`, 세션, FastAPI 배선)은 주인이 없으면 아무도 안 짠다.
> 설계상 세션/라우팅을 Interviewer 가 관리하므로 **담당 C 가 같이** 맡는다.

## 폴더 구조

```
src/interview/
├── config.py          # 환경변수/설정 (모두 여기서 import)
├── schemas/           # ⭐ 공용 계약 — 모두가 의존
│   ├── events.py      #   Interviewer 입력 이벤트 (음성/채팅 통합)
│   ├── evidence.py    #   EvidenceChunk + 메타데이터
│   ├── question.py    #   Question
│   ├── signals.py     #   AnswerQualitySignal (Assessment→Interviewer)
│   └── report.py      #   AnswerEvaluation / CompetencyModel / FinalReport
├── llm/               # 공용 LLM 래퍼 (모델/재시도 한곳 관리)
├── evidence/          # [A] indexing/sources/extract/chunking/store/retrieval
├── strategy/          # [B] agent/question_gen/difficulty/prompts/state
├── interviewer/       # [C] agent/graph/session/adapters
├── assessment/        # [D] agent/evaluator/report_builder/prompts
└── api/               # [C] FastAPI 진입점
```

### 각 패키지가 자라날 지도 (지금은 핵심만, 필요할 때 쪼갠다)

파일은 **역할이 명확히 다를 때만** 나눈다. 한 파일이 200~300줄을 넘거나
성격이 다른 일(LLM 호출 / 프롬프트 / 상태관리)이 섞이면 그때 분리.

- **evidence** (가장 큼): `sources`(MCP 접근) · `extract`(추출+메타) · `chunking` · `store`(vector DB) · `retrieval`(공용 툴) · `indexing`(파이프라인 배선)
- **strategy**: `agent`(진입점) · `question_gen`(LLM 생성) · `difficulty`(순수 규칙) · `prompts` · `state`
- **interviewer**: `agent`(라우팅) · `graph`(LangGraph 배선) · `session`(상태) · `adapters`(모드→공통 이벤트)
- **assessment**: `agent`(진입점) · `evaluator`(채점) · `report_builder` · `prompts`

> `prompts.py` 는 거의 모든 에이전트에 생긴다 — 프롬프트를 로직과 섞으면 튜닝이 지옥이라 처음부터 분리.

## 시작하기

[uv](https://docs.astral.sh/uv/) 사용 (가상환경 + 잠금파일 자동 관리):

```bash
# uv 설치 후
uv sync --extra dev          # 의존성 설치 (.venv 자동 생성)
cp .env.example .env         # 키 채우기 (.env 는 절대 커밋 금지)

uv run uvicorn interview.api.main:app --reload   # 서버 실행
uv run pytest                                    # 테스트
uv run ruff check .                              # 린트
```

의존성 추가 시: `uv add <패키지>` → `pyproject.toml` + `uv.lock` 갱신.
다른 팀원은 `uv sync` 한 번으로 동일 환경 ("내 컴퓨터에선 됐는데" 방지).

## 협업 순서 (권장)

1. **(Day 1, 다 같이)** `schemas/` 계약 확정.
2. **(Day 1~2, 다 같이)** 각 에이전트를 **가짜 데이터 반환 스텁**으로 만들어
   그래프가 끝에서 끝까지 한 바퀴 돌게 한다 (전부 가짜지만 연결됨).
3. **(이후, 각자)** 자기 폴더 내용물을 채운다. `NotImplementedError` /
   `TODO(담당 X)` 주석이 채울 자리 표시.
4. 통합을 마지막에 몰아서 하지 않는다 — 2번에서 이미 연결돼 있으니 점진 통합.

## 데이터 흐름 요약

```
[면접 전] api/index → evidence.build_index → evidence_store 구축
[면접 중] api/events → adapters(모드→이벤트) → InterviewerAgent.handle
            ├─ Assessment.evaluate → AnswerQualitySignal
            ├─ 신호로 라우팅 → Strategy.next_question / follow_up / hint
            └─ (Strategy/Assessment 는 필요시 search_evidence 로 근거 조회)
[종료]    Assessment.finalize → FinalReport (강점/약점/보완주제/학습추천)
```
