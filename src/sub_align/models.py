from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Cue:
    """A single subtitle cue with times in seconds."""

    index: int
    text: str
    start: float
    end: float
