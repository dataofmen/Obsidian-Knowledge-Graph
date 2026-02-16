"""CLI interface for Obsidian Knowledge Graph.

Commands: init, ingest, search, related, watch, stats
"""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.table import Table

from obsidian_kg.config import Config

console = Console()


def _run(coro):
    """Run an async coroutine from sync CLI context."""
    return asyncio.run(coro)


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """kg - Obsidian Knowledge Graph CLI"""
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.from_env()


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize Neo4j indices and constraints."""
    from obsidian_kg.graph_builder import init_graph

    config = ctx.obj["config"]
    console.print("\n🔧 Initializing knowledge graph...")
    console.print(f"   Neo4j: {config.neo4j_uri}")

    _run(init_graph(config))
    console.print("\n✅ Initialization complete!\n")


@cli.command()
@click.option("--folder", default=None, help="Specific folder to index (e.g., Projects)")
@click.option("--limit", default=None, type=int, help="Max number of notes to ingest")
@click.option("--concurrency", "-c", default=3, type=int, help="Parallel workers (default: 3)")
@click.pass_context
def ingest(ctx: click.Context, folder: str | None, limit: int | None, concurrency: int) -> None:
    """Ingest Obsidian notes into the knowledge graph."""
    from obsidian_kg.graph_builder import ingest_notes
    from obsidian_kg.parser import scan_vault

    config = ctx.obj["config"]
    scope = f"folder '{folder}'" if folder else "entire vault"
    console.print(f"\n📥 Ingesting {scope}...")
    console.print(f"   Vault: {config.vault_path}")

    notes = scan_vault(config.vault_path, folder=folder)

    if limit:
        notes = notes[:limit]

    console.print(f"   Found {len(notes)} notes (×{concurrency} workers)\n")

    stats = _run(ingest_notes(config, notes, concurrency=concurrency))

    table = Table(title="Ingestion Results")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("Ingested", f"[green]{stats['ingested']}[/green]")
    table.add_row("Skipped", f"[yellow]{stats['skipped']}[/yellow]")
    table.add_row("Errors", f"[red]{stats['errors']}[/red]")
    console.print(table)
    console.print()


@cli.command()
@click.argument("query")
@click.option("--num", default=10, help="Number of results")
@click.pass_context
def search(ctx: click.Context, query: str, num: int) -> None:
    """Search the knowledge graph."""
    from obsidian_kg.graph_builder import search_graph

    config = ctx.obj["config"]
    console.print(f"\n🔍 Searching: [bold]{query}[/bold]\n")

    results = _run(search_graph(config, query, num_results=num))

    if not results:
        console.print("  No results found.\n")
        return

    for i, r in enumerate(results, 1):
        score = f" (score: {r['score']:.3f})" if r.get("score") else ""
        console.print(f"  [{i}]{score} {r['fact']}")
        if r.get("source"):
            console.print(f"      [dim]{r['source']}[/dim]")
        console.print()


@cli.command()
@click.argument("filename")
@click.pass_context
def related(ctx: click.Context, filename: str) -> None:
    """Find notes related to a specific file."""
    from obsidian_kg.graph_builder import search_graph

    config = ctx.obj["config"]
    console.print(f"\n🔗 Related to: [bold]{filename}[/bold]\n")

    results = _run(search_graph(config, filename, num_results=10))

    if not results:
        console.print("  No related content found.\n")
        return

    for i, r in enumerate(results, 1):
        console.print(f"  [{i}] {r['fact']}")
        if r.get("source"):
            console.print(f"      [dim]{r['source']}[/dim]")
        console.print()


@cli.command()
@click.pass_context
def watch(ctx: click.Context) -> None:
    """Watch the vault for file changes and re-index automatically."""
    from obsidian_kg.watcher import watch_vault

    config = ctx.obj["config"]
    watch_vault(config)


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show knowledge graph statistics."""
    from obsidian_kg.graph_builder import get_graph_stats

    config = ctx.obj["config"]
    console.print("\n📊 Knowledge Graph Statistics\n")

    data = _run(get_graph_stats(config))

    table = Table()
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total Nodes", str(data["nodes"]))
    table.add_row("Total Edges", str(data["edges"]))
    table.add_row("Island Nodes", str(data["island_nodes"]))
    console.print(table)

    if data.get("labels"):
        console.print("\n  Labels:")
        for label, count in data["labels"].items():
            console.print(f"    {label}: {count}")
    console.print()


@cli.command()
@click.option("--port", default=8000, help="Server port")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.pass_context
def web(ctx: click.Context, port: int, host: str) -> None:
    """Launch the web dashboard UI."""
    import uvicorn

    console.print(f"\n🌐 Starting Knowledge Graph Dashboard...")
    console.print(f"   → http://localhost:{port}\n")

    uvicorn.run(
        "obsidian_kg.web:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    cli()
