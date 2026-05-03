"""MetricsAggregator for async training.

Accumulates metric observations from multiple sources (buffer, training loop,
coordinator) and reduces them with per-key aggregation rules at flush time.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any, Literal

import numpy as np

MetricRule = Literal["mean", "sum", "min", "max", "last", "std"]

# Keys that should be summed rather than averaged.
_SUM_KEYS: set[str] = {
    "groups/num_trajs_before_filter",
    "groups/num_trajs_after_filter",
    "groups/num_groups",
    "groups/dropped_min_trajs",
    "groups/dropped_zero_adv",
}

_LAST_PREFIXES: tuple[str, ...] = ("progress/",)


def _infer_rule(key: str) -> MetricRule:
    """Infer aggregation rule from metric key name.

    Unknown scalar metrics default to a mean. Counts/totals are summed,
    min/max are reduced by extrema, progress-like state uses last value.
    """
    if key in _SUM_KEYS:
        return "sum"

    for prefix in _LAST_PREFIXES:
        if key.startswith(prefix):
            return "last"
    if key.startswith("time/"):
        return "sum"

    parts = [part for part in key.lower().replace(":", "/").split("/") if part]
    leaf = parts[-1] if parts else key.lower()
    if leaf in {"min", "minimum"}:
        return "min"
    if leaf in {"max", "maximum"}:
        return "max"
    if leaf in {"sum", "total", "count", "counts", "num", "n", "tokens", "sequences"}:
        return "sum"
    if leaf.startswith("num_") or leaf.endswith(("_count", "_counts", "_tokens", "_sequences")):
        return "sum"
    if any(part in {"count", "counts", "total"} for part in parts):
        return "sum"

    return "mean"


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, np.number):
        return float(value)
    if hasattr(value, "numel") and hasattr(value, "detach") and hasattr(value, "item"):
        try:
            if value.numel() == 1:
                return float(value.detach().item())
        except Exception:
            return None
    return None


@dataclass
class _MetricState:
    rule: MetricRule
    total: float = 0.0
    weight: float = 0.0
    total_sq: float = 0.0
    value: float | None = None


class MetricsAggregator:
    """Accumulates metric observations and flushes as an aggregated plain dict.

    Usage::

        agg = MetricsAggregator()

        # record from various sources
        agg.record("episode/queue_wait", 0.3)
        agg.record("episode/queue_wait", 0.5)
        agg.record_dict(transform_metrics)

        # at log time
        plain_dict = agg.flush()  # reduces, clears, returns dict
    """

    def __init__(self) -> None:
        self._states: dict[str, _MetricState] = {}

    def _state(self, key: str, rule: MetricRule) -> _MetricState:
        state = self._states.get(key)
        if state is None:
            state = _MetricState(rule=rule)
            self._states[key] = state
        elif state.rule != rule:
            raise ValueError(f"Metric {key!r} recorded as both {state.rule!r} and {rule!r}")
        return state

    def record(self, key: str, value: Any, *, rule: MetricRule | None = None, weight: float = 1.0) -> None:
        """Record a single metric observation."""
        value = _to_float(value)
        if value is None:
            return

        rule = rule or _infer_rule(key)
        state = self._state(key, rule)
        if rule == "mean":
            state.total += value * float(weight)
            state.weight += float(weight)
        elif rule == "sum":
            state.total += value
        elif rule == "min":
            state.value = value if state.value is None else min(state.value, value)
        elif rule == "max":
            state.value = value if state.value is None else max(state.value, value)
        elif rule == "last":
            state.value = value
        elif rule == "std":
            state.total += value
            state.total_sq += value * value
            state.weight += 1.0

    def record_dict(self, metrics: dict) -> None:
        """Record all numeric values from a dict, coercing types."""
        for k, v in metrics.items():
            self.record(k, v)

    def record_distribution(self, prefix: str, values: list[float], *, fraction_zero: bool = False) -> None:
        for value in values:
            self.record(f"{prefix}/mean", value, rule="mean")
            self.record(f"{prefix}/std", value, rule="std")
            self.record(f"{prefix}/min", value, rule="min")
            self.record(f"{prefix}/max", value, rule="max")
            if fraction_zero:
                self.record(f"{prefix}/fraction_zero", 1.0 if abs(value) < 1e-8 else 0.0, rule="mean")

    def flush(self) -> dict[str, float]:
        """Reduce all accumulated values and return a plain dict. Clears state."""
        result = {}
        for key, state in self._states.items():
            if state.rule == "mean":
                result[key] = state.total / state.weight if state.weight > 0 else 0.0
            elif state.rule == "sum":
                result[key] = state.total
            elif state.rule == "std":
                if state.weight > 0:
                    mean = state.total / state.weight
                    variance = max(state.total_sq / state.weight - mean * mean, 0.0)
                    result[key] = float(np.sqrt(variance))
            elif state.rule in {"min", "max", "last"} and state.value is not None:
                result[key] = state.value
        self._states.clear()
        return result
