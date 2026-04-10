"""Centralized LLM factory — single place to construct attacker, target, and judge LLMs.

Supports multiple providers via config.yaml:
  attacker.provider: groq | openai | ollama | custom
  judge.provider:    anthropic | openai | custom

For "custom" provider, set:
  - ATTACKER_API_KEY env var for the attacker
  - JUDGE_API_KEY env var for the judge
  - base_url in the respective config section

The target has no provider field — it uses a plain OpenAI-compatible base_url/api_key
in config.yaml (already generic).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from redteamagentloop.config import AppConfig, TargetConfig


def build_attacker_llm(config: "AppConfig") -> BaseChatModel:
    """Build the attacker LLM from config."""
    from langchain_openai import ChatOpenAI

    ac = config.attacker

    if ac.provider == "ollama":
        api_key = "ollama"
    elif ac.provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY", "")
    elif ac.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
    else:  # custom
        api_key = os.environ.get("ATTACKER_API_KEY", "")

    return ChatOpenAI(
        model=ac.model,
        base_url=ac.base_url,
        api_key=api_key,
        temperature=ac.temperature,
        max_tokens=ac.max_tokens,
    )


def build_target_llm(target: "TargetConfig") -> BaseChatModel:
    """Build a target LLM from a TargetConfig."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=target.model,
        base_url=target.base_url,
        api_key=target.api_key,
        timeout=target.timeout_seconds,
        temperature=0.0,
    )


def build_judge_llm(config: "AppConfig") -> BaseChatModel:
    """Build the judge LLM from config.

    Returns a BaseChatModel. The caller is responsible for calling
    .with_structured_output(JudgeOutput, method="json_mode") before invoking.
    json_mode works for both Anthropic and OpenAI-compatible providers.
    """
    jc = config.judge

    if jc.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=jc.model,
            temperature=jc.temperature,
            max_tokens=jc.max_tokens,
        )

    # openai or custom — any OpenAI-compatible endpoint
    from langchain_openai import ChatOpenAI
    if jc.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
    else:  # custom
        api_key = os.environ.get("JUDGE_API_KEY", "")

    return ChatOpenAI(
        model=jc.model,
        base_url=jc.base_url,
        api_key=api_key,
        temperature=jc.temperature,
        max_tokens=jc.max_tokens,
    )
