"""Model registry: loads model definitions from YAML."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("nanoma")

# Default context limit for unknown models (128K is standard for modern LLMs)
_DEFAULT_CONTEXT_LIMIT = 128_000

# Fallback pricing — imported from cost.py for consistency
from nanoma.cost import _FALLBACK_PRICE_INPUT, _FALLBACK_PRICE_CACHED, _FALLBACK_PRICE_OUTPUT


@dataclass
class ModelDef:
    name: str
    provider: str
    context_limit: int
    price_input: float       # $/1M tokens (non-cached)
    price_cached: float      # $/1M tokens (cached)
    price_output: float      # $/1M tokens
    tier: str                # cheap, mid, strong


class ModelRegistry:
    def __init__(self):
        self.models: dict[str, ModelDef] = {}

    def load(self, path: Path | str):
        with open(path) as f:
            data = yaml.safe_load(f)
        for name, cfg in data.get("models", {}).items():
            p = cfg.get("pricing", {})
            model = ModelDef(
                name=name,
                provider=cfg.get("provider", "unknown"),
                context_limit=cfg.get("context_limit", _DEFAULT_CONTEXT_LIMIT),
                price_input=p.get("input", _FALLBACK_PRICE_INPUT),
                price_cached=p.get("cached_input", _FALLBACK_PRICE_CACHED),
                price_output=p.get("output", _FALLBACK_PRICE_OUTPUT),
                tier=cfg.get("tier", "mid"),
            )
            self.models[name] = model
            for alias in self._aliases_for(name):
                self.models.setdefault(alias, model)

    @staticmethod
    def _aliases_for(name: str) -> set[str]:
        aliases = {name}
        no_tier = name.split(":", 1)[0]
        aliases.add(no_tier)
        if "/" in no_tier:
            aliases.add(no_tier.rsplit("/", 1)[-1])
        return {alias for alias in aliases if alias}

    def get(self, name: str) -> ModelDef | None:
        return self.models.get(name)

    def pricing(self, name: str) -> tuple[float, float, float]:
        m = self.models.get(name)
        return (m.price_input, m.price_cached, m.price_output) if m else (
            _FALLBACK_PRICE_INPUT, _FALLBACK_PRICE_CACHED, _FALLBACK_PRICE_OUTPUT
        )

    def context_limit(self, name: str) -> int:
        m = self.models.get(name)
        return m.context_limit if m else _DEFAULT_CONTEXT_LIMIT

    def route(self, budget: float, allowed: list[str] | None = None) -> str:
        """Pick strongest model that fits budget.

        Estimates cost for a typical agent run (~10 turns, ~3000 input + 1500 output tokens/turn)
        and picks the most capable model whose estimated total cost stays under 50% of budget
        (reserving the rest for sub-agents or additional turns).
        """
        candidates = [m for m in self.models.values() if not allowed or m.name in allowed]
        if not candidates:
            return next(iter(self.models), "gpt-4o-mini")
        candidates.sort(key=lambda m: m.price_output)

        # Estimate: 10 turns × (3000 input + 1500 output) tokens per turn
        est_turns = 10
        est_input_per_turn = 3000
        est_output_per_turn = 1500
        budget_fraction = 0.5  # only use half the budget (leave room for spawned agents)

        for model in reversed(candidates):  # try strongest (most expensive) first
            est_cost = est_turns * (
                est_input_per_turn * model.price_input +
                est_output_per_turn * model.price_output
            ) / 1_000_000
            if est_cost < budget * budget_fraction:
                return model.name
        # Fall back to cheapest
        return candidates[0].name


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        default = Path(__file__).parent.parent / "models.yaml"
        if default.exists():
            _registry.load(default)
    return _registry


def load_models(path: Path | str) -> ModelRegistry:
    global _registry
    _registry = ModelRegistry()
    _registry.load(path)
    return _registry
