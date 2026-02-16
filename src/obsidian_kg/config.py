"""Configuration management for Obsidian Knowledge Graph.

Handles LLM provider selection (Ollama Cloud primary, OpenRouter fallback)
and Neo4j/Vault path settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class LLMProvider:
    """Configuration for an OpenAI-compatible LLM provider."""

    name: str
    api_key: str
    base_url: str
    model: str
    small_model: str


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "obsidian_kg_2024"

    # Obsidian Vault
    vault_path: Path = field(default_factory=lambda: Path.home())

    # LLM providers (ordered by priority)
    llm_providers: list[LLMProvider] = field(default_factory=list)

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> Config:
        """Load configuration from .env file and environment variables."""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        providers: list[LLMProvider] = []

        # Primary: OpenRouter (confirmed working)
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if openrouter_key:
            providers.append(
                LLMProvider(
                    name="openrouter",
                    api_key=openrouter_key,
                    base_url=os.getenv(
                        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                    ),
                    model=os.getenv(
                        "OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"
                    ),
                    small_model=os.getenv(
                        "OPENROUTER_SMALL_MODEL", "anthropic/claude-3-haiku"
                    ),
                )
            )

        # Fallback: Ollama Cloud
        ollama_key = os.getenv("OLLAMA_CLOUD_API_KEY", "")
        if ollama_key:
            providers.append(
                LLMProvider(
                    name="ollama-cloud",
                    api_key=ollama_key,
                    base_url=os.getenv(
                        "OLLAMA_CLOUD_BASE_URL", "https://api.ollama.com/v1"
                    ),
                    model=os.getenv("OLLAMA_CLOUD_MODEL", "llama3.1:70b"),
                    small_model=os.getenv(
                        "OLLAMA_CLOUD_SMALL_MODEL", "llama3.1:8b"
                    ),
                )
            )

        vault_path_str = os.getenv("OBSIDIAN_VAULT_PATH", "")
        vault_path = Path(vault_path_str) if vault_path_str else Path.home()

        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "obsidian_kg_2024"),
            vault_path=vault_path,
            llm_providers=providers,
        )

    @property
    def primary_llm(self) -> LLMProvider | None:
        """Get the primary (first available) LLM provider."""
        return self.llm_providers[0] if self.llm_providers else None

    @property
    def fallback_llm(self) -> LLMProvider | None:
        """Get the fallback (second) LLM provider."""
        return self.llm_providers[1] if len(self.llm_providers) > 1 else None
