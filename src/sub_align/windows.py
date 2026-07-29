from __future__ import annotations

from sub_align.models import Cue


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _speech_timeline(
    speech_spans: list[tuple[float, float]],
) -> tuple[list[tuple[float, float, float]], float]:
    """Return (segments of local_offset, global_start, duration), total_speech."""
    timeline: list[tuple[float, float, float]] = []
    cursor = 0.0
    for start, end in speech_spans:
        duration = max(0.0, end - start)
        if duration <= 0:
            continue
        timeline.append((cursor, start, duration))
        cursor += duration
    return timeline, cursor


def _map_speech_time(
    speech_t: float,
    timeline: list[tuple[float, float, float]],
    total_speech: float,
) -> float:
    if total_speech <= 0 or not timeline:
        return speech_t
    t = _clamp(speech_t, 0.0, total_speech)
    for local_offset, global_start, duration in timeline:
        if t <= local_offset + duration or local_offset + duration >= total_speech:
            return global_start + (t - local_offset)
    local_offset, global_start, duration = timeline[-1]
    return global_start + duration


def assign_windows(
    cues: list[Cue],
    *,
    mode: str = "realign",
    margin: float = 0.5,
    speech_spans: list[tuple[float, float]] | None = None,
    audio_duration: float | None = None,
) -> list[dict]:
    """
    Build WhisperX-style segments with search windows.

    - realign: map cues onto VAD speech spans by text length proportion
    - refine: keep original timestamps, expanded by margin
    """
    if not cues:
        return []
    if mode not in {"realign", "refine"}:
        raise ValueError(f"Unsupported mode: {mode!r}")

    if mode == "refine":
        segments: list[dict] = []
        for cue in cues:
            start = max(0.0, cue.start - margin)
            end = cue.end + margin
            if audio_duration is not None:
                end = min(end, audio_duration)
            if end <= start:
                end = start + 0.1
            segments.append({"text": cue.text, "start": start, "end": end})
        return segments

    spans = speech_spans or []
    if not spans:
        if audio_duration is None or audio_duration <= 0:
            raise ValueError("realign mode requires speech_spans or audio_duration")
        spans = [(0.0, audio_duration)]

    timeline, total_speech = _speech_timeline(spans)
    if total_speech <= 0:
        if audio_duration is None:
            raise ValueError("No speech detected and audio_duration is unknown")
        timeline, total_speech = _speech_timeline([(0.0, audio_duration)])

    weights = [max(len(cue.text.strip()), 1) for cue in cues]
    weight_sum = float(sum(weights))
    segments = []
    cursor = 0.0
    for cue, weight in zip(cues, weights, strict=True):
        share = total_speech * (weight / weight_sum)
        seg_start = _map_speech_time(cursor, timeline, total_speech)
        seg_end = _map_speech_time(cursor + share, timeline, total_speech)
        cursor += share
        start = max(0.0, seg_start - margin)
        end = seg_end + margin
        if audio_duration is not None:
            end = min(end, audio_duration)
        if end <= start:
            end = start + max(share, 0.1)
        segments.append({"text": cue.text, "start": start, "end": end})
    return segments
