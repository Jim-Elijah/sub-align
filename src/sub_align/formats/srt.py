from __future__ import annotations

import re
from pathlib import Path

from sub_align.models import Cue

# Wide-in: H:M:S with 1–2 digit fields; frac 1–6 digits; `,` or `.`
_TIME_RE = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[,.](?P<frac>\d{1,6}))?")
_ARROW_RE = re.compile(r"\s*-->\s*")


def _frac_to_seconds(frac: str | None) -> float:
    """Left-aligned fractional seconds: .5 / .50 / .500 → 0.5, .05 → 0.05."""
    if not frac:
        return 0.0
    padded = (frac + "000")[:3]
    return int(padded) / 1000.0


def _parse_timestamp(value: str) -> float:
    match = _TIME_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    return hours * 3600 + minutes * 60 + seconds + _frac_to_seconds(match.group("frac"))


def _format_timestamp(seconds: float) -> str:
    """Narrow-out: standard HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def loads(content: str) -> list[Cue]:
    """Parse SRT text into cues (tolerant timestamps; skip unusable blocks)."""
    blocks = re.split(r"\n\s*\n", content.strip(), flags=re.MULTILINE)
    cues: list[Cue] = []
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip() != ""]
        if len(lines) < 2:
            continue
        index_line = lines[0].strip()
        time_line = lines[1].strip()
        text = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
        if "-->" not in time_line:
            # Some files omit numeric index
            time_line = index_line
            text = "\n".join(lines[1:]).strip()
            index = len(cues) + 1
        else:
            try:
                index = int(index_line)
            except ValueError:
                index = len(cues) + 1
        parts = _ARROW_RE.split(time_line, maxsplit=1)
        if len(parts) != 2:
            continue
        start_raw, end_raw = parts[0].strip(), parts[1].strip()
        try:
            start = _parse_timestamp(start_raw)
            end = _parse_timestamp(end_raw)
        except ValueError:
            continue
        cues.append(Cue(index=index, text=text, start=start, end=end))
    return cues


def dumps(cues: list[Cue]) -> str:
    """Serialize cues to SRT text (standard HH:MM:SS,mmm).

    Cue numbers are always rewritten as 1..n in file order, regardless of
    ``cue.index`` on the input objects.
    """
    parts: list[str] = []
    for i, cue in enumerate(cues, start=1):
        parts.append(
            f"{i}\n"
            f"{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}\n"
            f"{cue.text}\n"
        )
    return "\n".join(parts).rstrip() + "\n"


def load(path: str | Path) -> list[Cue]:
    return loads(Path(path).read_text(encoding="utf-8"))


def dump(path: str | Path, cues: list[Cue]) -> None:
    for i, cue in enumerate(cues, start=1):
        cue.index = i
    Path(path).write_text(dumps(cues), encoding="utf-8")
