"""Ethical content filter for RedTeamAgentLoop.

Blocks prompts that touch CBRN, CSAM, or other absolute off-limits categories.
These categories are hardcoded and NOT configurable — they are bright lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Blocked categories — hardcoded, not user-configurable.
# ---------------------------------------------------------------------------

BLOCKED_CATEGORIES: list[str] = [
    # Child safety
    "CSAM",
    "child sexual abuse",
    "child pornography",
    "child exploitation",
    "minor sexual",
    # Chemical / biological / radiological / nuclear
    "CBRN",
    "bioweapon",
    "biological weapon",
    "chemical weapon",
    "nerve agent",
    "sarin",
    "VX gas",
    "mustard gas",
    "anthrax",
    "ricin",
    "nuclear weapon",
    "dirty bomb",
    "radiological weapon",
    "weaponize pathogen",
    # Synthesis of dangerous substances
    "synthesize fentanyl",
    "synthesize methamphetamine",
    "synthesize nerve",
    "how to make explosives",
    "instructions for bomb",
    # Self-harm
    "suicide method",
    "how to kill myself",
    "self-harm instructions",
]

# Pre-compile patterns for O(n) scanning with no repeated regex compilation.
_BLOCKED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(re.escape(term), re.IGNORECASE), term)
    for term in BLOCKED_CATEGORIES
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    allowed: bool
    blocked_category: str | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_prompt(prompt: str) -> GuardrailResult:
    """Scan *prompt* for blocked categories.

    Returns ``GuardrailResult(allowed=True)`` if the prompt is clean,
    ``GuardrailResult(allowed=False, ...)`` if a blocked term was matched.
    """
    for pattern, category in _BLOCKED_PATTERNS:
        if pattern.search(prompt):
            return GuardrailResult(
                allowed=False,
                blocked_category=category,
                reason=f"Prompt contains blocked category: '{category}'",
            )
    return GuardrailResult(allowed=True)
