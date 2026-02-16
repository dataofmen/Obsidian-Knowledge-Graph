"""File system watcher for Obsidian Vault changes.

Monitors .md file create/modify/delete events and triggers re-indexing.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from rich.console import Console
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from obsidian_kg.config import Config
from obsidian_kg.graph_builder import create_graphiti, ingest_notes
from obsidian_kg.parser import parse_note

logger = logging.getLogger(__name__)
console = Console()


class VaultEventHandler(FileSystemEventHandler):
    """Handles file system events for .md files in the Obsidian vault."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self._loop = asyncio.new_event_loop()

    def _is_markdown(self, path: str) -> bool:
        return path.endswith(".md") and not Path(path).name.startswith(".")

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_markdown(event.src_path):
            console.print(f"  📝 New: [green]{Path(event.src_path).name}[/green]")
            self._reindex(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_markdown(event.src_path):
            console.print(f"  ✏️  Modified: [yellow]{Path(event.src_path).name}[/yellow]")
            self._reindex(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_markdown(event.src_path):
            console.print(f"  🗑  Deleted: [red]{Path(event.src_path).name}[/red]")
            # Note: Graphiti doesn't easily support entity deletion.
            # Deleted notes are logged but not removed from the graph.
            logger.info("File deleted: %s (graph nodes preserved)", event.src_path)

    def _reindex(self, file_path: str) -> None:
        """Re-index a single changed file."""
        try:
            note = parse_note(Path(file_path))
            self._loop.run_until_complete(
                ingest_notes(self.config, [note])
            )
            console.print(f"  ✓ Re-indexed: [bold]{note.title}[/bold]")
        except Exception as e:
            logger.error("Failed to re-index %s: %s", file_path, e)
            console.print(f"  ✗ Error: {e}")


def watch_vault(config: Config) -> None:
    """Start watching the Obsidian vault for file changes.

    Blocks until interrupted with Ctrl+C.
    """
    vault_path = str(config.vault_path)
    handler = VaultEventHandler(config)
    observer = Observer()
    observer.schedule(handler, vault_path, recursive=True)

    console.print(f"\n👁  Watching: [bold]{vault_path}[/bold]")
    console.print("   Press Ctrl+C to stop.\n")

    observer.start()
    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        console.print("\n  Stopping watcher...")
    finally:
        observer.stop()
        observer.join()
