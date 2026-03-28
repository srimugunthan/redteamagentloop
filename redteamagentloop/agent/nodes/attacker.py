"""Attacker node — selects a strategy and generates an adversarial prompt."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from redteamagentloop.agent.strategies import STRATEGY_REGISTRY, AttackStrategy

if TYPE_CHECKING:
    from redteamagentloop.agent.state import RedTeamState

# Module-level strategy rotation index, protected by a lock.
_strategy_index: int = 0
_strategy_lock = asyncio.Lock()


def _available_strategies(failed: set[str]) -> list[str]:
    names = sorted(STRATEGY_REGISTRY.keys())
    available = [n for n in names if n not in failed]
    return available if available else names  # fallback: retry all if all failed


async def _next_strategy(failed: set[str]) -> str:
    global _strategy_index
    async with _strategy_lock:
        available = _available_strategies(failed)
        strategy = available[_strategy_index % len(available)]
        _strategy_index += 1
        return strategy


async def attacker_node(state: "RedTeamState", config: RunnableConfig) -> dict:
    if state["iteration_count"] >= state["max_iterations"]:
        return {"error": "max_iterations_reached"}

    cfg = config.get("configurable", {})
    app_config = cfg.get("app_config")
    storage_manager = cfg.get("storage_manager")

    # LLM — use injected override (tests) or build from config.
    attacker_llm = cfg.get("attacker_llm")
    if attacker_llm is None:
        ac = app_config.attacker
        attacker_llm = ChatOpenAI(
            model=ac.model,
            base_url=ac.base_url,
            api_key=os.environ["GROQ_API_KEY"],
            temperature=ac.temperature,
            max_tokens=ac.max_tokens,
        )

    # Strategy selection — rotate away from failed strategies.
    current = state["current_strategy"]
    failed = state["failed_strategies"]
    if not current or current in failed:
        current = await _next_strategy(failed)

    strategy: AttackStrategy = AttackStrategy.from_name(current)

    # Generate prompt with up to 3 retries on LLM error.
    prompt = ""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            # Use temperature bump on retry attempts to diversify output.
            if attempt > 0 and hasattr(attacker_llm, "temperature"):
                attacker_llm = attacker_llm.bind(
                    temperature=min(1.0, attacker_llm.temperature + 0.1 * attempt)
                )
            prompt = await strategy.generate_prompt(state, attacker_llm)
            if prompt:
                break
        except Exception as exc:
            last_error = exc
            continue

    if not prompt:
        return {"error": f"Attacker LLM failed after 3 attempts: {last_error}"}

    # Dedup check — if too similar to an existing prompt, try up to 2 more times.
    if storage_manager is not None:
        for _ in range(2):
            if not storage_manager._chroma.is_duplicate(prompt):
                break
            try:
                prompt = await strategy.generate_prompt(state, attacker_llm)
            except Exception:
                break

    return {
        "current_prompt": prompt,
        "current_strategy": current,
        "iteration_count": state["iteration_count"] + 1,
        "error": None,
    }
