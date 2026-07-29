from __future__ import annotations

from pathlib import Path

from sub_align.models import Cue


def loads(content: str) -> list[Cue]:
    """Parse plain text (one line per cue) into cues with zeroed timestamps."""
    cues: list[Cue] = []
    for i, raw in enumerate(content.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        cues.append(Cue(index=i, text=line, start=0.0, end=0.0))
    return cues


def load(path: str | Path) -> list[Cue]:
    return loads(Path(path).read_text(encoding="utf-8"))
