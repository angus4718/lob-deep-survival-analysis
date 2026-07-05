"""Latency providers for raw-feed backtests."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Generic, Protocol, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class LatencyResult(Generic[T]):
    """Callable result paired with an inference latency in nanoseconds."""

    value: T
    latency_ns: int


class LatencyProvider(Protocol):
    """Measure or supply latency for one strategy call."""

    def run(self, fn: Callable[[], T]) -> LatencyResult[T]:
        """Run ``fn`` and return its output plus the modeled latency."""


@dataclass(frozen=True)
class StaticLatencyProvider:
    """Use a configured latency regardless of actual model runtime."""

    latency_ns: int = 0

    @classmethod
    def from_seconds(cls, seconds: float) -> "StaticLatencyProvider":
        return cls(latency_ns=int(float(seconds) * 1e9))

    @classmethod
    def from_milliseconds(cls, milliseconds: float) -> "StaticLatencyProvider":
        return cls(latency_ns=int(float(milliseconds) * 1e6))

    @classmethod
    def from_microseconds(cls, microseconds: float) -> "StaticLatencyProvider":
        return cls(latency_ns=int(float(microseconds) * 1e3))

    def run(self, fn: Callable[[], T]) -> LatencyResult[T]:
        return LatencyResult(value=fn(), latency_ns=max(0, int(self.latency_ns)))


@dataclass(frozen=True)
class MeasuredLatencyProvider:
    """Use wall-clock runtime of the strategy call as modeled latency."""

    min_latency_ns: int = 0

    def run(self, fn: Callable[[], T]) -> LatencyResult[T]:
        start = perf_counter_ns()
        value = fn()
        elapsed = perf_counter_ns() - start
        return LatencyResult(value=value, latency_ns=max(int(self.min_latency_ns), elapsed))
