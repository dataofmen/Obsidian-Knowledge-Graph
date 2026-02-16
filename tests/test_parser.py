"""Tests for the Obsidian Vault parser."""

from pathlib import Path

import pytest

from obsidian_kg.parser import ObsidianNote, parse_note, scan_vault


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Create a temporary vault with test markdown files."""
    # Simple note with frontmatter, tags, and links
    note1 = tmp_path / "Projects" / "test-project.md"
    note1.parent.mkdir(parents=True)
    note1.write_text(
        """---
tags:
  - project
  - ai
---

# Test Project

This is a test project about [[Knowledge Graph]] and #machine-learning.

Related to [[Another Note]] and uses [[Python]].
""",
        encoding="utf-8",
    )

    # Note without frontmatter
    note2 = tmp_path / "Memo" / "quick-note.md"
    note2.parent.mkdir(parents=True)
    note2.write_text(
        """# Quick Note

Just a quick memo with #todo tag.
No links here.
""",
        encoding="utf-8",
    )

    # Korean note
    note3 = tmp_path / "Memo" / "한글노트.md"
    note3.write_text(
        """---
tags: [메모, 아이디어]
---

# 한글 테스트

이것은 #한글태그 로 된 테스트입니다.
[[프로젝트A]]와 연관된 내용입니다.
""",
        encoding="utf-8",
    )

    # Empty note
    note4 = tmp_path / "empty.md"
    note4.write_text("", encoding="utf-8")

    return tmp_path


def test_parse_note_with_frontmatter(tmp_vault: Path) -> None:
    """Frontmatter tags and body links should be extracted."""
    note = parse_note(tmp_vault / "Projects" / "test-project.md")

    assert note.title == "test-project"
    assert "project" in note.tags
    assert "ai" in note.tags
    assert "machine-learning" in note.tags
    assert "Knowledge Graph" in note.internal_links
    assert "Another Note" in note.internal_links
    assert "Python" in note.internal_links
    assert note.folder == "Projects"


def test_parse_note_without_frontmatter(tmp_vault: Path) -> None:
    """Notes without frontmatter should parse body tags."""
    note = parse_note(tmp_vault / "Memo" / "quick-note.md")

    assert note.title == "quick-note"
    assert "todo" in note.tags
    assert note.internal_links == []
    assert note.frontmatter == {}


def test_parse_korean_note(tmp_vault: Path) -> None:
    """Korean tags and links should be correctly extracted."""
    note = parse_note(tmp_vault / "Memo" / "한글노트.md")

    assert note.title == "한글노트"
    assert "한글태그" in note.tags
    assert "메모" in note.tags
    assert "아이디어" in note.tags
    assert "프로젝트A" in note.internal_links


def test_parse_empty_note(tmp_vault: Path) -> None:
    """Empty files should parse without errors."""
    note = parse_note(tmp_vault / "empty.md")

    assert note.title == "empty"
    assert note.body == ""
    assert note.tags == []
    assert note.internal_links == []


def test_scan_vault_all(tmp_vault: Path) -> None:
    """scan_vault should find all .md files."""
    notes = scan_vault(tmp_vault)
    assert len(notes) == 4


def test_scan_vault_folder(tmp_vault: Path) -> None:
    """scan_vault with folder filter should only return files from that folder."""
    notes = scan_vault(tmp_vault, folder="Memo")
    assert len(notes) == 2
    assert all(n.folder == "Memo" for n in notes)


def test_scan_vault_nonexistent_folder(tmp_vault: Path) -> None:
    """scan_vault with nonexistent folder should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        scan_vault(tmp_vault, folder="NonExistent")


def test_note_timestamps(tmp_vault: Path) -> None:
    """Parsed notes should have modification timestamps."""
    note = parse_note(tmp_vault / "Projects" / "test-project.md")
    assert note.modified_at is not None
