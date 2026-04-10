"""Configuration models and loader for RedTeamAgentLoop."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

class TargetConfig(BaseModel):
    model: str
    base_url: str
    api_key: str = "ollama"
    system_prompt: str = ""
    timeout_seconds: int = 30
    output_tag: str
    rpm: int = 0  # calls per minute; 0 = no limit


class AttackerConfig(BaseModel):
    provider: Literal["groq", "openai", "ollama", "custom"] = "groq"
    model: str
    base_url: str
    temperature: float = 0.9
    max_tokens: int = 1024
    rpm: int = 0  # calls per minute; 0 = no limit


class JudgeConfig(BaseModel):
    provider: Literal["anthropic", "openai", "custom"] = "anthropic"
    model: str
    base_url: str = ""  # required for openai/custom providers
    temperature: float = 0.1
    max_tokens: int = 512
    rpm: int = 0  # calls per minute; 0 = no limit


class LoopConfig(BaseModel):
    max_iterations: int = 50
    vuln_threshold: float = 7.0
    mutation_batch_size: int = 3
    strategy_rotation: bool = True
    max_mutations_per_strategy: int = 8  # rotate after this many mutation engine cycles
    early_stop_on_success: bool = False


class StorageConfig(BaseModel):
    jsonl_path: str = "reports/{target_tag}_vulnerabilities.jsonl"
    sqlite_path: str = "reports/metadata.db"
    chromadb_path: str = "reports/chroma_{target_tag}"
    dedup_threshold: float = 0.92
    embedding_model: str = "all-MiniLM-L6-v2"

    def jsonl_path_for(self, target_tag: str) -> str:
        return self.jsonl_path.format(target_tag=target_tag)

    def chromadb_path_for(self, target_tag: str) -> str:
        return self.chromadb_path.format(target_tag=target_tag)


class ReportingConfig(BaseModel):
    html_template: str = "reports/templates/report.html.j2"
    output_dir: str = "reports/output"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    targets: list[TargetConfig] = Field(min_length=1)
    attacker: AttackerConfig
    judge: JudgeConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @field_validator("attacker")
    @classmethod
    def warn_if_api_key_in_config(cls, v: AttackerConfig) -> AttackerConfig:
        _api_key_pattern = re.compile(r"sk-[a-zA-Z0-9]{32,}")
        for field_value in v.model_dump().values():
            if isinstance(field_value, str) and _api_key_pattern.search(field_value):
                print(
                    "WARNING: config.yaml appears to contain an API key. "
                    "Use environment variables instead.",
                    file=sys.stderr,
                )
                break
        return v


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_AUTHORIZATION_REQUIRED = "AUTHORIZED: true"


def check_authorization(auth_path: str = "authorization.txt") -> None:
    """Exit with a clear message if authorization.txt is missing or invalid."""
    path = Path(auth_path)
    if not path.exists():
        print(
            f"ERROR: {auth_path} not found. "
            "Create this file and acknowledge authorization before running.",
            file=sys.stderr,
        )
        sys.exit(1)
    if _AUTHORIZATION_REQUIRED not in path.read_text():
        print(
            f"ERROR: {auth_path} does not contain the required acknowledgment line:\n"
            f'  "{_AUTHORIZATION_REQUIRED}"',
            file=sys.stderr,
        )
        sys.exit(1)


def check_api_keys(app_config: "AppConfig") -> None:
    """Exit fast with a clear error if required API keys are missing from env.

    Called once at CLI startup, before any node runs, so failures are immediate
    rather than surfacing mid-loop on the first LLM call.
    """
    import os
    missing: list[str] = []
    if app_config.attacker.provider == "groq" and not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if app_config.attacker.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if app_config.attacker.provider == "custom" and not os.environ.get("ATTACKER_API_KEY"):
        missing.append("ATTACKER_API_KEY")
    if app_config.judge.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if app_config.judge.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if app_config.judge.provider == "custom" and not os.environ.get("JUDGE_API_KEY"):
        missing.append("JUDGE_API_KEY")
    if missing:
        print(
            f"ERROR: Required environment variable(s) not set: {', '.join(missing)}.\n"
            "Set them in your shell or in a .env file before running.",
            file=sys.stderr,
        )
        sys.exit(1)


def load_config(path: str = "config.yaml") -> AppConfig:
    """Parse config.yaml and return a validated AppConfig."""
    config_path = Path(path)
    if not config_path.exists():
        print(f"ERROR: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    raw = yaml.safe_load(config_path.read_text())
    return AppConfig.model_validate(raw)
