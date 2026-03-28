"""Attack strategy registry — import all strategies to trigger registration."""

from redteamagentloop.agent.strategies.base import STRATEGY_REGISTRY, AttackStrategy
from redteamagentloop.agent.strategies.finserv_specific import FinServSpecific
from redteamagentloop.agent.strategies.injection import (
    DirectInjection,
    FewShotPoisoning,
    IndirectInjection,
    NestedInstruction,
)
from redteamagentloop.agent.strategies.jailbreak import DirectJailbreak, PersonaHijack
from redteamagentloop.agent.strategies.obfuscation import (
    AdversarialSuffix,
    ContextOverflow,
    ObfuscatedRequest,
)

__all__ = [
    "STRATEGY_REGISTRY",
    "AttackStrategy",
    "DirectJailbreak",
    "PersonaHijack",
    "DirectInjection",
    "IndirectInjection",
    "FewShotPoisoning",
    "NestedInstruction",
    "AdversarialSuffix",
    "ContextOverflow",
    "ObfuscatedRequest",
    "FinServSpecific",
]
