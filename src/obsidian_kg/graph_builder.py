"""Knowledge graph builder using Graphiti.

Handles Graphiti initialization, episode ingestion from Obsidian notes,
and LLM provider fallback logic (Ollama Cloud → OpenRouter).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from obsidian_kg.config import Config, LLMProvider
from obsidian_kg.parser import ObsidianNote

logger = logging.getLogger(__name__)
console = Console()


def _create_llm_client(provider: LLMProvider) -> OpenAIGenericClient:
    """Create a Graphiti-compatible LLM client from a provider config."""
    config = LLMConfig(
        api_key=provider.api_key,
        base_url=provider.base_url,
        model=provider.model,
        small_model=provider.small_model,
    )
    return OpenAIGenericClient(config=config)


def _create_embedder(provider: LLMProvider) -> OpenAIEmbedder:
    """Create an embedder using the same provider credentials."""
    config = OpenAIEmbedderConfig(
        api_key=provider.api_key,
        base_url=provider.base_url,
    )
    return OpenAIEmbedder(config=config)


async def create_graphiti(config: Config) -> Graphiti:
    """Initialize Graphiti with Neo4j and LLM client.

    Tries primary provider first, falls back to secondary on failure.
    """
    primary = config.primary_llm
    fallback = config.fallback_llm

    if not primary:
        raise ValueError(
            "No LLM provider configured. "
            "Set OLLAMA_CLOUD_API_KEY or OPENROUTER_API_KEY in .env"
        )

    # Try primary provider
    try:
        llm_client = _create_llm_client(primary)
        embedder = _create_embedder(primary)
        graphiti = Graphiti(
            config.neo4j_uri,
            config.neo4j_user,
            config.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
        )
        console.print(f"  ✓ LLM: [green]{primary.name}[/green] ({primary.model})")
        return graphiti
    except Exception as e:
        logger.warning("Primary LLM (%s) failed: %s", primary.name, e)
        if not fallback:
            raise

    # Try fallback provider
    try:
        llm_client = _create_llm_client(fallback)
        embedder = _create_embedder(fallback)
        graphiti = Graphiti(
            config.neo4j_uri,
            config.neo4j_user,
            config.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
        )
        console.print(
            f"  ⚠ Primary LLM failed, using fallback: "
            f"[yellow]{fallback.name}[/yellow] ({fallback.model})"
        )
        return graphiti
    except Exception as e:
        raise RuntimeError(
            f"Both LLM providers failed. "
            f"Primary ({primary.name}): see logs. "
            f"Fallback ({fallback.name}): {e}"
        ) from e


async def init_graph(config: Config) -> None:
    """Initialize Neo4j indices and constraints for Graphiti."""
    graphiti = await create_graphiti(config)
    try:
        await graphiti.build_indices_and_constraints()
        console.print("  ✓ Neo4j indices and constraints created")
    finally:
        await graphiti.close()


async def ingest_notes(
    config: Config,
    notes: list[ObsidianNote],
    *,
    batch_size: int = 5,
    concurrency: int = 5,
) -> dict[str, int]:
    """Ingest Obsidian notes as episodes into the knowledge graph.

    Args:
        config: Application config.
        notes: List of parsed ObsidianNote objects.
        batch_size: (Unused, kept for API compat.)
        concurrency: Max concurrent ingestion tasks.

    Returns:
        Dict with counts: {"ingested": N, "skipped": N, "errors": N}
    """
    stats = {"ingested": 0, "skipped": 0, "errors": 0}
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def _ingest_one(note: ObsidianNote, progress, task_id):
        async with semaphore:
            try:
                content = _build_episode_content(note)
                ref_time = note.modified_at or datetime.now()

                # Each task creates its own Graphiti to avoid shared-state
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

                async with lock:
                    stats["ingested"] += 1

            except Exception as e:
                logger.error("Failed to ingest %s: %s", note.title, e)
                async with lock:
                    stats["errors"] += 1

            progress.update(task_id, advance=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            f"Ingesting {len(notes)} notes (×{concurrency} workers)...",
            total=len(notes),
        )

        tasks = [_ingest_one(note, progress, task_id) for note in notes]
        await asyncio.gather(*tasks)

    return stats


def _build_episode_content(note: ObsidianNote) -> str:
    """Build a structured episode content string from an Obsidian note."""
    parts = [f"# {note.title}\n"]

    if note.tags:
        parts.append(f"Tags: {', '.join(note.tags)}")

    if note.internal_links:
        parts.append(f"Related notes: {', '.join(note.internal_links)}")

    if note.frontmatter:
        # Include relevant frontmatter keys
        for key, value in note.frontmatter.items():
            if key not in ("tags",) and value:
                parts.append(f"{key}: {value}")

    parts.append(f"\n{note.body}")

    return "\n".join(parts)


async def search_graph(
    config: Config,
    query: str,
    *,
    num_results: int = 10,
) -> list[dict]:
    """Search the knowledge graph using hybrid search.

    Args:
        config: Application config.
        query: Search query string.
        num_results: Maximum results to return.

    Returns:
        List of result dicts with entity/edge information.
    """
    graphiti = await create_graphiti(config)
    try:
        results = await graphiti.search(query, num_results=num_results)

        # Extract basic facts from edges
        edge_data = []
        episode_ids = set()
        for r in results:
            ep_ids = getattr(r, "episodes", None) or getattr(r, "episode_ids", None) or []
            episode_ids.update(ep_ids)
            edge_data.append({
                "fact": r.fact,
                "score": getattr(r, "score", None),
                "source": getattr(r, "source_description", None),
                "created_at": str(getattr(r, "created_at", "")),
                "episodes": list(ep_ids)[:3],  # keep first few
            })

        # Look up episode source_descriptions from Neo4j
        if episode_ids:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                config.neo4j_uri,
                auth=(config.neo4j_user, config.neo4j_password),
            )
            try:
                with driver.session() as session:
                    ep_records = session.run(
                        "MATCH (e:Episodic) WHERE e.uuid IN $ids "
                        "RETURN e.uuid AS id, e.source_description AS source, "
                        "e.name AS name",
                        ids=list(episode_ids),
                    ).data()

                ep_map = {r["id"]: r for r in ep_records}

                # Enrich edge data with episode source
                for item in edge_data:
                    if not item["source"]:
                        for ep_id in item.get("episodes", []):
                            ep = ep_map.get(ep_id)
                            if ep and ep.get("source"):
                                item["source"] = ep["source"]
                                break
                        # Fallback: use episode name
                        if not item["source"]:
                            for ep_id in item.get("episodes", []):
                                ep = ep_map.get(ep_id)
                                if ep and ep.get("name"):
                                    item["source"] = f"Obsidian note: {ep['name']}"
                                    break
            finally:
                driver.close()

        return edge_data
    finally:
        await graphiti.close()


async def get_graph_stats(config: Config) -> dict:
    """Get statistics about the knowledge graph."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )

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
