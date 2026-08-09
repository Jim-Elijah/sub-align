from __future__ import annotations

import re
from pathlib import Path

from sub_align.models import Cue

# Wide-in: minutes 1–3 digits, seconds 1–2, frac 1–6 digits, `.` or `,`
_TAG_RE = re.compile(r"\[(?P<m>\d{1,3}):(?P<s>\d{1,2})(?:[.,](?P<frac>\d{1,6}))?\]\s*(?P<text>.*)")
_DEFAULT_DURATION = 3.0


def _frac_to_seconds(frac: str | None) -> float:
    """Left-aligned fractional seconds: .5 / .50 / .500 → 0.5, .05 → 0.05."""
    if not frac:
        return 0.0
    padded = (frac + "000")[:3]
    return int(padded) / 1000.0


def _parse_timestamp(match: re.Match[str]) -> float:
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    return minutes * 60 + seconds + _frac_to_seconds(match.group("frac"))


def _format_timestamp(seconds: float) -> str:
    """Narrow-out: standard [mm:ss.xx] centiseconds."""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    minutes, rem = divmod(total_cs, 6000)
    secs, cs = divmod(rem, 100)
    return f"[{minutes:02d}:{secs:02d}.{cs:02d}]"


def loads(
    content: str,
    default_duration: float = _DEFAULT_DURATION,
    *,
    audio_duration: float | None = None,
) -> list[Cue]:
    """Parse LRC text into cues. End times come from the next cue start.

    The last cue ends at ``min(start + default_duration, audio_duration)`` when
    ``audio_duration`` is provided and greater than ``start``.
    """
    timed: list[tuple[float, str]] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("[ti:") or line.startswith("[ar:"):
            continue
        if line.startswith("[al:") or line.startswith("[by:") or line.startswith("[offset:"):
            continue
        match = _TAG_RE.fullmatch(line)
        if not match:
            continue
        text = match.group("text").strip()
        if text == "":
            continue
        timed.append((_parse_timestamp(match), text))

    cues: list[Cue] = []
    for i, (start, text) in enumerate(timed):
        if i + 1 < len(timed):
            end = timed[i + 1][0]
            if end <= start:
                end = start + default_duration
        else:
            end = start + default_duration
            if audio_duration is not None and audio_duration > start:
                end = min(end, audio_duration)
        cues.append(Cue(index=i + 1, text=text, start=start, end=end))
    return cues


def dumps(cues: list[Cue]) -> str:
    """Serialize cues to LRC text (start timestamps only, standard format).

    LRC has no index field; a round-trip via ``loads`` assigns 1..n.
    """
    lines = [_format_timestamp(cue.start) + cue.text for cue in cues]
    return "\n".join(lines) + ("\n" if lines else "")


def load(
    path: str | Path,
    default_duration: float = _DEFAULT_DURATION,
    *,
    audio_duration: float | None = None,
) -> list[Cue]:
    return loads(
        Path(path).read_text(encoding="utf-8"),
        default_duration=default_duration,
        audio_duration=audio_duration,
    )


def dump(path: str | Path, cues: list[Cue]) -> None:
    for i, cue in enumerate(cues, start=1):
        cue.index = i
    Path(path).write_text(dumps(cues), encoding="utf-8")
