"""Lightweight step timing for pipeline stages.

Usage::

    timings = Timings(enabled=True)  # prints to stderr
    with timings.step("transcribe"):
        ...
    timings.finish()
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TextIO


@dataclass(frozen=True)
class TimingRecord:
    name: str
    seconds: float


@dataclass
class Timings:
    """Collect named step durations and optionally print them."""

    enabled: bool = True
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    prefix: str = "[timing]"
    records: list[TimingRecord] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter, init=False)

    def _emit(self, name: str, seconds: float) -> None:
        self.records.append(TimingRecord(name, seconds))
        if self.enabled:
            print(
                f"{self.prefix} {name}: {seconds:.2f}s",
                file=self.stream,
                flush=True,
            )

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Time a block. When disabled, still yields with negligible overhead."""
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._emit(name, time.perf_counter() - start)

    def finish(self, name: str = "total") -> float:
        """Emit wall time since construction. Returns elapsed seconds."""
        elapsed = time.perf_counter() - self._t0
        if self.enabled:
            self._emit(name, elapsed)
        return elapsed
