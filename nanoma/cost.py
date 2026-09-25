"""CostLedger: budget tracking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("nanoma")

# Fallback pricing ($/1M tokens) when model is not in registry.
# Intentionally conservative (overestimates) to avoid budget overruns.
_FALLBACK_PRICE_INPUT = 1.0    # $/1M input tokens
_FALLBACK_PRICE_CACHED = 0.1   # $/1M cached input tokens
_FALLBACK_PRICE_OUTPUT = 3.0   # $/1M output tokens


@dataclass
class UsageRecord:
    """Token usage from a single LLM call."""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self) -> float:
        from nanoma.models import get_registry
        try:
            pi, pc, po = get_registry().pricing(self.model)
        except Exception:
            logger.debug(f"Using fallback pricing for unknown model: {self.model}")
            pi, pc, po = _FALLBACK_PRICE_INPUT, _FALLBACK_PRICE_CACHED, _FALLBACK_PRICE_OUTPUT
        non_cached = max(0, self.input_tokens - self.cached_input_tokens)
        return (non_cached * pi + self.cached_input_tokens * pc + self.output_tokens * po) / 1_000_000


@dataclass
class CostLedger:
    """Global cost tracking."""
    total_budget: float = 10.0
    total_spent: float = 0.0
    per_agent: dict[str, float] = field(default_factory=dict)

    def remaining(self) -> float:
        return max(0.0, self.total_budget - self.total_spent)

    def can_afford(self, cost: float) -> bool:
        return self.remaining() >= cost

    def record(self, agent_id: str, usage: UsageRecord) -> float:
        cost = usage.cost_usd()
        self.total_spent += cost
        self.per_agent[agent_id] = self.per_agent.get(agent_id, 0.0) + cost
        return cost

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int = 1000) -> float:
        from nanoma.models import get_registry
        try:
            pi, _, po = get_registry().pricing(model)
        except Exception:
            pi, po = _FALLBACK_PRICE_INPUT, _FALLBACK_PRICE_OUTPUT
        return (input_tokens * pi + output_tokens * po) / 1_000_000

    def summary(self) -> dict[str, Any]:
        return {
            "budget": self.total_budget,
            "spent": round(self.total_spent, 6),
            "remaining": round(self.remaining(), 6),
            "per_agent": {k: round(v, 6) for k, v in self.per_agent.items()},
        }
