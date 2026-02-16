"""Obsidian Vault markdown parser.

Parses .md files extracting frontmatter, internal links, tags, and body text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import frontmatter

# Patterns for Obsidian-specific syntax
INTERNAL_LINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z\uac00-\ud7a3][\w/\-\uac00-\ud7a3]*)", re.UNICODE)


@dataclass
class ObsidianNote:
    """Parsed representation of an Obsidian markdown note."""

    title: str
    file_path: Path
    body: str
    frontmatter: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    internal_links: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    modified_at: datetime | None = None

    @property
    def folder(self) -> str:
        """Get the immediate parent folder name."""
        return self.file_path.parent.name

    @property
    def relative_path(self) -> str:
        """Get path relative to vault root (set externally)."""
        return str(self.file_path)


def parse_note(file_path: Path) -> ObsidianNote:
    """Parse a single Obsidian markdown file into an ObsidianNote.

    Args:
        file_path: Absolute path to the .md file.

    Returns:
        ObsidianNote with extracted metadata.
    """
    text = file_path.read_text(encoding="utf-8")
    post = frontmatter.loads(text)

    body = post.content
    fm = dict(post.metadata) if post.metadata else {}

    # Extract internal links [[Target]] or [[Target|Alias]]
    internal_links = INTERNAL_LINK_PATTERN.findall(body)

    # Extract tags #tag (supports Korean, nested tags like #tag/subtag)
    tags_from_body = TAG_PATTERN.findall(body)
    tags_from_fm = fm.get("tags", [])
    if isinstance(tags_from_fm, str):
        tags_from_fm = [tags_from_fm]
    all_tags = list(set(tags_from_body + tags_from_fm))

    # File timestamps
    stat = file_path.stat()
    created_at = datetime.fromtimestamp(stat.st_birthtime) if hasattr(stat, "st_birthtime") else None
    modified_at = datetime.fromtimestamp(stat.st_mtime)

    return ObsidianNote(
        title=file_path.stem,
        file_path=file_path,
        body=body,
        frontmatter=fm,
        tags=all_tags,
        internal_links=internal_links,
        created_at=created_at,
        modified_at=modified_at,
    )


def scan_vault(vault_path: Path, folder: str | None = None) -> list[ObsidianNote]:
    """Scan an Obsidian vault directory and parse all markdown files.

    Args:
        vault_path: Root path of the Obsidian vault.
        folder: Optional subfolder to scan (e.g., "Projects").

    Returns:
        List of parsed ObsidianNote objects.
    """
    search_path = vault_path / folder if folder else vault_path
    if not search_path.exists():
        raise FileNotFoundError(f"Path does not exist: {search_path}")

    notes: list[ObsidianNote] = []
    for md_file in sorted(search_path.rglob("*.md")):
        # Skip dotfiles and hidden directories
        if any(part.startswith(".") for part in md_file.relative_to(vault_path).parts):
            continue
        try:
            note = parse_note(md_file)
            notes.append(note)
        except Exception as e:
            # Log but don't crash on malformed files
            print(f"  ⚠ Skipped {md_file.name}: {e}")

    return notes
