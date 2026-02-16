# Obsidian Knowledge Graph 🧠🕸️

Obsidian Vault의 정적인 노트를 시간 인지형(temporal-aware) 지식 그래프로 변환하고, AI 어시스턴트와 연결하여 입체적인 지식 관리 환경을 제공하는 오픈소스 프로젝트입니다.

---

## 🎯 기획 의도 (Project Intent)

대부분의 개인 지식 관리(PKM) 도구는 정보가 축적될수록 원하는 내용을 찾기 힘들어지고, 지식 간의 유기적인 연결을 놓치기 쉽습니다. 이 프로젝트는 다음 세 가지 목표를 달성하고자 기획되었습니다:

1.  **지식의 연결성 극대화**: 단순한 폴더/태그 구조를 넘어, AI가 노트 간의 숨겨진 맥락적 관계를 파악하여 그래프 형태로 시각화합니다.
2.  **시간 흐름의 인지**: 각 노트가 작성되고 수정된 시간을 기록하여, 지식이 시간에 따라 어떻게 진화하고 연결되는지 추적합니다.
3.  **지능형 지식 검색(RAG)**: 내 지식 기반 위에서 AI가 직접 답변하도록 하여, 기억나지 않는 세부 정보를 즉시 복구하고 지식을 재구성합니다.

---

## ⌚️ 시간 기반 지식 그래프 (Temporal Knowledge Graph)

이 프로젝트의 핵심은 정보를 단순히 나열하는 것이 아니라, **시간의 흐름**에 따라 관리한다는 점입니다.

### 🌟 왜 필요한가요? (사진 앨범 vs 실시간 CCTV)
- **기존 방식 (사진)**: "홍길동은 팀장이다"라는 정적 사실만 저장합니다. 나중에 역할이 바뀌어도 과거 데이터와 충돌이 발생합니다.
- **시간 기반 방식 (CCTV)**: "2월에는 팀장이었고, 3월부터는 본부장이 되었다"와 같이 지식의 **유효 기간**을 관리합니다. 이를 통해 지식의 진화 과정을 추적할 수 있습니다.

### 🛠️ 어떻게 구현했나요?
1.  **에피소드(Episode) 기반 기록**: 각 노트를 하나의 독립된 사건으로 취급합니다.
2.  **기준 시간(Reference Time) 추출**: 노트를 읽을 때 파일의 수정 시간이나 본문 내 날짜를 분석하여 지식의 '발생 시점'을 확정합니다.
3.  **동적 유효성(valid_at/expired_at)**: Neo4j 그래프에 지식을 저장할 때 시간 정보를 함께 기록하여, AI가 가장 최신의 정보나 특정 시점의 맥락을 이해하고 답변할 수 있게 합니다.

---

## 🛠️ 기술 스택 (Tech Stack)

| 구분 | 기술 / 라이브러리 | 설명 |
| :--- | :--- | :--- |
| **언어** | Python 3.10+ | 고성능 지식 추출 및 서버 로직 구현 |
| **패키지 관리** | [uv](https://github.com/astral-sh/uv) | 차세대 Rust 기반 Python 패키지 매니저 |
| **DB** | **Neo4j 5.26+** | 엔터티 기반 지식 그래프 저장 및 벡터 검색 지원 |
| **LLM 엔진** | **Graphiti** | 대규모 지식 추출 및 동적 그래프 구축 코어 |
| **백엔드** | FastAPI | 고성능 REST API 서버 및 비동기 작업 처리 |
| **프런트엔드** | Vanilla JS + D3.js | 가볍고 빠른 그래프 시각화 및 대시보드 UI |
| **인프라** | Docker Compose | Neo4j 및 서비스 환경의 간편한 구축 |

---

## 🏗️ 시스템 아키텍처 (Architecture)

1.  **Parser**: Obsidian Vault를 스캔하여 Markdown, Frontmatter, Internal Links를 정밀하게 분석합니다.
2.  **Graphiti-Core**: LLM(GPT-4o, Llama 3.1 등)을 활용해 텍스트에서 'Episode'를 생성하고, 개체(Entity)와 관계(Relation)를 지식 그래프로 변환합니다.
3.  **Vector Store**: Neo4j의 벡터 인덱스를 사용하여 질문과 가장 유사한 지식 조각을 1단계로 검색합니다.
4.  **Graph Traversal**: 검색된 개체와 연결된 주변 지식을 그래프 횡단(Traversal)을 통해 가져와 풍부한 답변 맥락을 형성합니다.
5.  **Web Dashboard**: 실시간 인제스트 제어, 통계 확인, 그래프 탐색 및 AI 채팅 인터페이스를 제공합니다.

---

## 📖 문서

- [활용 가이드 (Usage Guide)](./USAGE_GUIDE.md) — 구체적인 활용 사례와 AI 대화 예시
- [개발 가이드 (Development Guide)](./docs/DEVELOPMENT.md)

---

## 🚀 시작하기

```bash
# 1. .env 설정
cp .env.example .env
# Edit .env with your API keys (OpenRouter or Ollama Cloud)

# 2. Neo4j 실행 (Docker)
docker compose up -d

# 3. 개발 환경 구축 및 설치
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 4. 그래프 초기화 및 웹 서버 실행
kg init
uv run kg web
```

---

## Commands

| Command | Description |
| :--- | :--- |
| `kg init` | Neo4j 인덱스 및 제약 조건 초기 설정 |
| `kg ingest` | Vault의 모든 노트를 지식 그래프로 변환 (중복 스킵 가능) |
| `kg watch` | Obsidian 파일 변경을 실시간으로 감시하여 그래프 자동 업데이트 |
| `kg web` | 대화형 웹 대시보드 서버 (Port 8000) 실행 |
| `kg stats` | 현재 구축된 그래프의 통계(노드, 엣지 수 등) 조회 |
