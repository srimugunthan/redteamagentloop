"""State schema for RedTeamAgentLoop.

All nodes receive a RedTeamState and return a partial dict of changed fields.
LangGraph merges those partials using the reducers defined here.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


# ---------------------------------------------------------------------------
# Reducer functions
# ---------------------------------------------------------------------------

def append_to_list(existing: list, new: list) -> list:
    """Append new items to the existing list rather than replacing it.

    Used for attack_history and successful_attacks so that each node's
    partial update accumulates rather than overwrites.
    """
    return existing + new


def union_sets(existing: set[str], new: set[str]) -> set[str]:
    """Union two sets — used for failed_strategies so exhausted strategies
    accumulate across iterations without duplication.
    """
    return existing | new


# ---------------------------------------------------------------------------
# AttackRecord — one completed attack attempt
# ---------------------------------------------------------------------------

class AttackRecord(TypedDict):
    """Snapshot of a single attack iteration."""
    session_id: str
    iteration: int
    strategy: str
    prompt: str
    response: str
    score: float
    score_rationale: str
    timestamp: str          # ISO-8601
    was_successful: bool    # True if score >= vuln_threshold
    mutation_depth: int     # 0 = original prompt, N = Nth mutation


# ---------------------------------------------------------------------------
# RedTeamState — full live state of one agent run
# ---------------------------------------------------------------------------

class RedTeamState(TypedDict):
    # --- Current turn ---
    current_strategy: str
    current_prompt: str
    current_response: str
    score: float
    score_rationale: str
    iteration_count: int

    # --- Memory (append-only via reducers) ---
    attack_history: Annotated[list[AttackRecord], append_to_list]
    successful_attacks: Annotated[list[AttackRecord], append_to_list]
    failed_strategies: Annotated[set[str], union_sets]

    # --- Mutation pipeline ---
    mutation_queue: list[str]       # prompts queued for retry with mutations
    current_mutations: list[str]    # mutations generated in this cycle

    # --- Run configuration (set once at start, never mutated) ---
    target_system_prompt: str
    target_objective: str           # what the target must NOT do
    max_iterations: int
    vuln_threshold: float
    session_id: str

    # --- Error signalling ---
    error: str | None
