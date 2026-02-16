# Obsidian Knowledge Graph

Obsidian Vault에서 시간 인지(temporal-aware) 지식 그래프를 구축하는 AI 업무 보조 시스템.

- **Web Dashboard**: Interactive graph visualization, entity search, and real-time ingestion management.
- **AI Chat (RAG)**: Chat with your knowledge graph to summarize concepts or find connections.
- **Auto-Sync Optimization**: Automatically skips already ingested notes to save time and API costs.

## 📖 문서

- [활용 가이드 (Usage Guide)](./USAGE_GUIDE.md) — 구체적인 활용 사례와 AI 대화 예시
- [개발 가이드 (Development Guide)](./docs/DEVELOPMENT.md)

## 🚀 시작하기

```bash
# 1. .env 설정
cp .env.example .env
# Edit .env with your API keys

# 2. Neo4j 시작
docker compose up -d

# 3. 의존성 설치
uv venv && uv pip install -e ".[dev]"

# 4. 그래프 초기화
uv run kg init

# 5. Vault 인덱싱
uv run kg ingest

# 6. 검색
uv run kg search "지식 그래프"
```

## Commands

| Command | Description |
|---------|-------------|
| `kg init` | Neo4j 인덱스/제약조건 초기 설정 |
| `kg ingest` | Vault 전체 인덱싱 |
| `kg search "query"` | 하이브리드 검색 |
| `kg related "file"` | 관련 컨텍스트 조회 |
| `kg watch` | 실시간 파일 변경 감시 |
| `kg stats` | 그래프 통계 |
