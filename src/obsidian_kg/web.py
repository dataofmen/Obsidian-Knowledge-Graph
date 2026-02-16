"""FastAPI web server for Knowledge Graph dashboard.

Provides REST API endpoints and serves the interactive web UI.
Includes AI chat with RAG-based knowledge graph context.
"""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from obsidian_kg.config import Config

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config on startup."""
    app.state.config = Config.from_env()
    yield


app = FastAPI(title="Obsidian Knowledge Graph", lifespan=lifespan)


# ── Static files ──────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main dashboard page."""
    return FileResponse(STATIC_DIR / "index.html")


# ── Helpers ───────────────────────────────────────────────
def _get_neo4j_driver(config: Config):
    """Create a sync Neo4j driver."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )


def _source_to_obsidian_uri(source: str | None, vault_path: Path) -> dict:
    """Convert source_description to file path and obsidian:// URI.

    source_description format: "Obsidian note: Folder/Title"
    """
    if not source:
        return {"file_path": None, "obsidian_uri": None, "vault_relative": None}

    # Parse "Obsidian note: Folder/Title"
    match = re.match(r"Obsidian note:\s*(.+)", source)
    if not match:
        return {"file_path": None, "obsidian_uri": None, "vault_relative": None}

    relative = match.group(1).strip()
    # Try to find the actual file
    md_path = vault_path / f"{relative}.md"

    # Build obsidian:// URI (vault name is the last folder name)
    vault_name = vault_path.name
    obsidian_uri = f"obsidian://open?vault={vault_name}&file={relative}"

    return {
        "file_path": str(md_path) if md_path.exists() else None,
        "obsidian_uri": obsidian_uri,
        "vault_relative": relative,
    }


def _run_neo4j_stats(config: Config) -> dict:
    """Run stats query synchronously."""
    driver = _get_neo4j_driver(config)
    try:
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edge_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS c"
            ).single()["c"]
            island_count = session.run(
                "MATCH (n) WHERE NOT (n)--() RETURN count(n) AS c"
            ).single()["c"]
            label_counts = session.run(
                "MATCH (n) UNWIND labels(n) AS label "
                "RETURN label, count(*) AS c ORDER BY c DESC"
            ).data()

        return {
            "nodes": node_count,
            "edges": edge_count,
            "island_nodes": island_count,
            "labels": {r["label"]: r["c"] for r in label_counts},
        }
    finally:
        driver.close()


def _run_neo4j_graph(config: Config, limit: int, focus: str | None = None) -> dict:
    """Run graph query synchronously, optionally focusing on a specific node."""
    driver = _get_neo4j_driver(config)
    try:
        with driver.session() as session:
            if focus:
                # Fetch node by name AND its direct relationships
                results = session.run(
                    "MATCH (a:Entity {name: $focus})-[r:RELATES_TO]-(b:Entity) "
                    "WHERE r.expired_at IS NULL "
                    "RETURN a.uuid AS a_id, a.name AS a_name, a.summary AS a_summary, labels(a) AS a_labels, "
                    "b.uuid AS b_id, b.name AS b_name, b.summary AS b_summary, labels(b) AS b_labels, "
                    "r.name AS r_label, r.fact AS r_fact, r.created_at AS r_created_at "
                    "ORDER BY r.created_at DESC LIMIT $limit",
                    focus=focus,
                    limit=limit,
                ).data()
            else:
                # Fetch recent RELATES_TO edges and the nodes they connect
                results = session.run(
                    "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                    "WHERE r.expired_at IS NULL "
                    "RETURN a.uuid AS a_id, a.name AS a_name, a.summary AS a_summary, labels(a) AS a_labels, "
                    "b.uuid AS b_id, b.name AS b_name, b.summary AS b_summary, labels(b) AS b_labels, "
                    "r.name AS r_label, r.fact AS r_fact, r.created_at AS r_created_at "
                    "ORDER BY r.created_at DESC LIMIT $limit",
                    limit=limit,
                ).data()

        nodes_map = {}
        links = []

        for row in results:
            # Add source node
            if row["a_id"] not in nodes_map:
                nodes_map[row["a_id"]] = {
                    "id": row["a_id"],
                    "name": row["a_name"] or "?",
                    "summary": row["a_summary"] or "",
                    "group": row["a_labels"][0] if row["a_labels"] else "Entity",
                }
            # Add target node
            if row["b_id"] not in nodes_map:
                nodes_map[row["b_id"]] = {
                    "id": row["b_id"],
                    "name": row["b_name"] or "?",
                    "summary": row["b_summary"] or "",
                    "group": row["b_labels"][0] if row["b_labels"] else "Entity",
                }
            # Add link
            links.append({
                "source": row["a_id"],
                "target": row["b_id"],
                "label": row["r_label"] or "",
                "fact": row["r_fact"] or "",
            })

        # Fallback: if no edges, just get some nodes
        if not links:
            nodes_data = session.run(
                "MATCH (n:Entity) RETURN n.uuid AS id, n.name AS name, "
                "n.summary AS summary, labels(n) AS labels LIMIT $limit",
                limit=limit
            ).data()
            nodes = [
                {
                    "id": n["id"],
                    "name": n["name"] or "?",
                    "summary": n["summary"] or "",
                    "group": n["labels"][0] if n["labels"] else "Entity",
                }
                for n in nodes_data
            ]
            return {"nodes": nodes, "links": []}

        return {"nodes": list(nodes_map.values()), "links": links}
    finally:
        driver.close()


def _run_neo4j_entities(config: Config, limit: int) -> dict:
    """Run entities query synchronously."""
    driver = _get_neo4j_driver(config)
    try:
        with driver.session() as session:
            data = session.run(
                "MATCH (n:Entity) "
                "OPTIONAL MATCH (n)-[r:RELATES_TO]->() "
                "WITH n, count(r) AS connections "
                "RETURN n.uuid AS id, n.name AS name, "
                "n.summary AS summary, connections "
                "ORDER BY connections DESC LIMIT $limit",
                limit=limit,
            ).data()
        return {"entities": data, "count": len(data)}
    finally:
        driver.close()


def _run_neo4j_ingested_sources(config: Config) -> set[str]:
    """Run ingested sources query synchronously."""
    driver = _get_neo4j_driver(config)
    try:
        with driver.session() as session:
            results = session.run("MATCH (e:Episodic) RETURN e.name AS name").data()
            return {r["name"] for r in results if r.get("name")}
    finally:
        driver.close()


# ── API: Statistics ───────────────────────────────────────
@app.get("/api/stats")
async def api_stats():
    """Return graph statistics."""
    config: Config = app.state.config
    try:
        stats = await asyncio.to_thread(_run_neo4j_stats, config)
        return stats
    except Exception as e:
        logger.error("Stats API error: %s", e)
        return JSONResponse(
            {"error": "Neo4j 연결 실패", "detail": str(e)},
            status_code=503,
        )


# ── API: Search (enhanced with document links) ───────────
@app.get("/api/search")
async def api_search(
    q: str = Query(..., min_length=1, description="Search query"),
    num: int = Query(10, ge=1, le=50),
):
    """Hybrid search with document links."""
    from obsidian_kg.graph_builder import search_graph

    config: Config = app.state.config
    try:
        results = await search_graph(config, q, num_results=num)

        # Enrich results with document links
        for r in results:
            link_info = _source_to_obsidian_uri(r.get("source"), config.vault_path)
            r.update(link_info)

        return {"query": q, "results": results, "count": len(results)}
    except Exception as e:
        logger.error("Search API error: %s", e)
        return JSONResponse(
            {"error": "검색 실패", "detail": str(e)},
            status_code=503,
        )


# ── API: Graph data (D3.js format) ────────────────────────
@app.get("/api/graph")
async def api_graph(
    limit: int = Query(200, ge=10, le=1000),
    focus: str | None = Query(None, description="Entity name to center graph around")
):
    """Return nodes and edges for D3.js force graph visualization."""
    config: Config = app.state.config
    try:
        data = await asyncio.to_thread(_run_neo4j_graph, config, limit, focus)
        return data
    except Exception as e:
        logger.error("Graph API error: %s", e)
        return JSONResponse(
            {"error": "그래프 데이터 로드 실패", "detail": str(e)},
            status_code=503,
        )


# ── API: Entity list ──────────────────────────────────────
@app.get("/api/entities")
async def api_entities(limit: int = Query(100, ge=1, le=500)):
    """Return entity list with summaries."""
    config: Config = app.state.config
    try:
        data = await asyncio.to_thread(_run_neo4j_entities, config, limit)
        return data
    except Exception as e:
        logger.error("Entities API error: %s", e)
        return JSONResponse(
            {"error": "엔티티 목록 로드 실패", "detail": str(e)},
            status_code=503,
        )


# ── API: AI Chat (RAG) ───────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """AI assistant using knowledge graph as context (RAG).

    1. Search KG for relevant facts
    2. Build context from facts + document snippets
    3. Send to LLM for synthesis
    4. Return answer with source citations
    """
    from obsidian_kg.graph_builder import search_graph

    config: Config = app.state.config
    provider = config.primary_llm

    if not provider:
        return JSONResponse({"error": "LLM 설정 없음"}, status_code=503)

    try:
        # 1. Retrieve relevant context from knowledge graph
        kg_results = await search_graph(config, req.message, num_results=8)

        # Build context string with source citations
        context_parts = []
        sources = []
        for i, r in enumerate(kg_results, 1):
            fact = r.get("fact", "")
            source = r.get("source", "")
            context_parts.append(f"[{i}] {fact}")
            if source:
                link_info = _source_to_obsidian_uri(source, config.vault_path)
                sources.append({
                    "index": i,
                    "fact": fact,
                    "source": source,
                    **link_info,
                })

        context = "\n".join(context_parts)

        # 2. Build system prompt for RAG
        system_prompt = (
            "당신은 사용자의 Obsidian 지식그래프를 기반으로 답변하는 AI 어시스턴트입니다.\n"
            "아래 지식그래프에서 검색된 관련 정보를 참고하여 답변하세요.\n"
            "답변할 때 출처가 되는 정보의 번호 [1], [2] 등을 인용해주세요.\n"
            "지식그래프에 없는 내용은 추측하지 말고, 없다고 알려주세요.\n\n"
            f"=== 지식그래프 검색 결과 ===\n{context}\n=========================\n"
        )

        # 3. Build conversation messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add history (last 6 turns)
        for h in req.history[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": req.message})

        # 4. Call LLM via OpenAI-compatible API
        import httpx

        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{provider.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {provider.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": provider.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        answer = data["choices"][0]["message"]["content"]

        return {
            "answer": answer,
            "sources": sources,
            "model": provider.model,
            "context_count": len(kg_results),
        }

    except Exception as e:
        logger.error("Chat API error: %s", e)
        return JSONResponse(
            {"error": "AI 응답 실패", "detail": str(e)},
            status_code=503,
        )


# ── API: Read document content ────────────────────────────
@app.get("/api/document")
async def api_document(path: str = Query(...)):
    """Read and return document content for preview."""
    config: Config = app.state.config
    file_path = Path(path)

    # Security: ensure the path is within the vault
    try:
        file_path.resolve().relative_to(config.vault_path.resolve())
    except ValueError:
        return JSONResponse({"error": "잘못된 경로"}, status_code=403)

    if not file_path.exists():
        return JSONResponse({"error": "파일을 찾을 수 없습니다"}, status_code=404)

    try:
        content = file_path.read_text(encoding="utf-8")
        return {
            "title": file_path.stem,
            "content": content,
            "folder": file_path.parent.name,
            "path": str(file_path),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Ingestion State ───────────────────────────────────────
class IngestionState:
    """Global ingestion progress tracker."""

    def __init__(self):
        self.running = False
        self.total = 0
        self.completed = 0
        self.skipped = 0
        self.errors = 0
        self.current_note = ""
        self.task: asyncio.Task | None = None

    def to_dict(self):
        pct = (
            int(((self.completed + self.skipped) / self.total * 100))
            if self.total > 0
            else 0
        )
        return {
            "running": self.running,
            "total": self.total,
            "completed": self.completed,
            "skipped": self.skipped,
            "errors": self.errors,
            "current_note": self.current_note,
            "percent": pct,
        }


ingest_state = IngestionState()


# ── API: Vault Info ───────────────────────────────────────
@app.get("/api/vault/info")
async def api_vault_info():
    """Return vault path and note count."""
    config: Config = app.state.config
    vault_path = config.vault_path

    if not vault_path.exists():
        return {"path": str(vault_path), "exists": False, "note_count": 0}

    from obsidian_kg.parser import scan_vault

    try:
        notes = scan_vault(vault_path)
        return {
            "path": str(vault_path),
            "exists": True,
            "note_count": len(notes),
            "vault_name": vault_path.name,
        }
    except Exception as e:
        return {"path": str(vault_path), "exists": True, "note_count": 0, "error": str(e)}


class VaultConfigRequest(BaseModel):
    vault_path: str


@app.post("/api/vault/config")
async def api_vault_config(req: VaultConfigRequest):
    """Update vault path (runtime only — also update .env for persistence)."""
    import os

    new_path = Path(req.vault_path).expanduser().resolve()
    if not new_path.exists():
        return JSONResponse({"error": f"경로가 존재하지 않습니다: {new_path}"}, status_code=400)

    app.state.config.vault_path = new_path
    os.environ["OBSIDIAN_VAULT_PATH"] = str(new_path)

    # Try to update .env file for persistence
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        try:
            lines = env_file.read_text().splitlines()
            updated = False
            for i, line in enumerate(lines):
                if line.startswith("OBSIDIAN_VAULT_PATH="):
                    lines[i] = f"OBSIDIAN_VAULT_PATH={new_path}"
                    updated = True
                    break
            if not updated:
                lines.append(f"OBSIDIAN_VAULT_PATH={new_path}")
            env_file.write_text("\n".join(lines) + "\n")
        except Exception:
            pass  # non-critical

    return {"path": str(new_path), "vault_name": new_path.name}


# ── API: Ingestion management ─────────────────────────────
# ── Ingestion Helpers ─────────────────────────────────────
async def _get_ingested_sources() -> set[str]:
    """Return set of Obsidian note titles already in Neo4j."""
    config: Config = app.state.config
    try:
        sources = await asyncio.to_thread(_run_neo4j_ingested_sources, config)
        return sources
    except Exception as e:
        logger.error("Failed to fetch ingested sources: %s", e)
        return set()


class IngestRequest(BaseModel):
    concurrency: int = 5
    folder: str | None = None


@app.post("/api/ingest/start")
async def api_ingest_start(req: IngestRequest):
    """Start background ingestion."""
    global ingest_state

    if ingest_state.running:
        return JSONResponse({"error": "인제스트가 이미 실행 중입니다"}, status_code=409)

    config: Config = app.state.config

    from obsidian_kg.parser import scan_vault

    notes = scan_vault(config.vault_path, folder=req.folder)
    if not notes:
        return JSONResponse({"error": "인제스트할 노트가 없습니다"}, status_code=404)

    # Filter out already ingested notes
    existing_titles = await _get_ingested_sources()
    to_ingest = [n for n in notes if n.title not in existing_titles]
    skipped_count = len(notes) - len(to_ingest)

    ingest_state = IngestionState()
    ingest_state.running = True
    ingest_state.total = len(notes)
    ingest_state.skipped = skipped_count

    if not to_ingest:
        ingest_state.running = False
        return {
            "status": "already_done",
            "total": len(notes),
            "skipped": skipped_count,
        }

    async def _run_ingestion():
        """Background ingestion task with per-note progress tracking."""
        from obsidian_kg.graph_builder import create_graphiti, _build_episode_content

        sem = asyncio.Semaphore(req.concurrency)

        async def _ingest_one(note):
            async with sem:
                if not ingest_state.running:
                    return  # cancelled
                ingest_state.current_note = note.title
                try:
                    content = _build_episode_content(note)
                    ref_time = note.modified_at or __import__("datetime").datetime.now()
                    graphiti = await create_graphiti(config)
                    try:
                        await graphiti.add_episode(
                            name=note.title,
                            episode_body=content,
                            source_description=f"Obsidian note: {note.folder}/{note.title}",
                            reference_time=ref_time,
                        )
                    finally:
                        await graphiti.close()
                    ingest_state.completed += 1
                except Exception as e:
                    logger.error("Ingest error (%s): %s", note.title, e)
                    ingest_state.errors += 1
                    # Note: we don't increment completed here if it really failed? 
                    # Actually, for progress bar, we should treat it as 'done with error'
                    ingest_state.completed += 1

        try:
            tasks = [_ingest_one(note) for note in to_ingest]
            await asyncio.gather(*tasks)
        finally:
            ingest_state.running = False
            ingest_state.current_note = ""

    ingest_state.task = asyncio.create_task(_run_ingestion())

    return {
        "status": "started",
        "total": len(notes),
        "to_ingest": len(to_ingest),
        "skipped": skipped_count,
        "concurrency": req.concurrency,
    }


@app.get("/api/ingest/status")
async def api_ingest_status():
    """Return current ingestion progress."""
    return ingest_state.to_dict()


@app.post("/api/ingest/stop")
async def api_ingest_stop():
    """Stop running ingestion."""
    global ingest_state

    if not ingest_state.running:
        return {"status": "not_running"}

    ingest_state.running = False  # Signal workers to stop
    if ingest_state.task:
        ingest_state.task.cancel()
        try:
            await ingest_state.task
        except asyncio.CancelledError:
            pass

    return {
        "status": "stopped",
        "completed": ingest_state.completed,
        "total": ingest_state.total,
    }


# ── Runner ────────────────────────────────────────────────
def main():
    """Run the web server."""
    import uvicorn

    uvicorn.run(
        "obsidian_kg.web:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
