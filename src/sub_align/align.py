from __future__ import annotations

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

from sub_align.audio import audio_duration, load_audio, trim_audio
from sub_align.device import default_compute_type, resolve_device
from sub_align.formats import lrc, srt, txt
from sub_align.models import Cue
from sub_align.vad import detect_speech_spans
from sub_align.windows import assign_windows

# WhisperX may split one input cue into multiple sentence segments; remap via
# alphanumeric tokens so contractions like "I'm" stay consistent across sides.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MIN_WINDOW = 0.1
# Floor for .txt ASR search windows so short lines are not force-aligned alone
# inside a near-zero span.
_MIN_SEARCH_DURATION = 0.5
# Filled / oversized search windows longer than this are narrowed onto VAD
# speech so forced alignment does not dump a short cue at the next utterance.
_MAX_SEARCH_WINDOW = 4.0
# Ignore speech islands that only touch the right edge of a fill gap (usually
# the start of the next matched cue).
_GAP_EDGE_GUARD = 0.5
# Cues at or below this token count are temporarily merged with neighbors for
# WhisperX alignment, then split back via word remapping.
_SHORT_CUE_TOKENS = 3
# Aligned cue shorter than this is treated as failed and repaired from windows.
_MIN_ALIGNED_DURATION = 0.05
# Global offset for .srt/.lrc: need enough ASR↔cue matches; ignore tiny drift.
_MIN_OFFSET_MATCHES = 3
_MIN_ABS_OFFSET = 0.25
# Suggested limits when callers opt into length/duration splitting (CLI/API).
_TRANSCRIPT_MAX_WORDS = 12
_TRANSCRIPT_MAX_CHARS = 42
_TRANSCRIPT_MAX_DURATION = 8.0
_EN_STRONG_BREAKS = ".?!;"
_GENERIC_STRONG_BREAKS = "。？！；"
_SECONDARY_BREAKS = ",:，：、"


def _read_cues(
    path: Path,
    *,
    audio_duration: float | None = None,
) -> list[Cue]:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return srt.load(path)
    if suffix == ".lrc":
        return lrc.load(path, audio_duration=audio_duration)
    if suffix == ".txt":
        return txt.load(path)
    raise ValueError(f"Unsupported subtitle format: {suffix}")


def _write_cues(path: Path, cues: list[Cue]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        srt.dump(path, cues)
        return
    if suffix == ".lrc":
        lrc.dump(path, cues)
        return
    raise ValueError(f"Unsupported subtitle format: {suffix}")


def _cues_from_transcription(segments: list[dict]) -> list[Cue]:
    cues: list[Cue] = []
    for idx, segment in enumerate(segments, start=1):
        start_raw = segment.get("start")
        end_raw = segment.get("end")
        if start_raw is None or end_raw is None:
            continue
        start = float(start_raw)
        end = float(end_raw)
        if end < start:
            end = start
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        cues.append(Cue(index=idx, text=text, start=start, end=end))
    return cues


def _split_sentence_like_units(text: str, language: str | None) -> list[str]:
    """Split text by strong punctuation first (English + generic fallback)."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []
    strong = _EN_STRONG_BREAKS + _GENERIC_STRONG_BREAKS
    if language and language.lower().startswith("en"):
        strong = _EN_STRONG_BREAKS + _GENERIC_STRONG_BREAKS
    pattern = rf"(?<=[{re.escape(strong)}])\s+"
    parts = [part.strip() for part in re.split(pattern, cleaned) if part.strip()]
    return parts or [cleaned]


def _split_by_length(
    text: str,
    *,
    max_words: int | None = None,
    max_chars: int | None = None,
) -> list[str]:
    """Length fallback split: prefer boundaries on light punctuation."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_words is None and max_chars is None:
        return [cleaned]
    words = cleaned.split()
    if not words:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    for word in words:
        candidate = " ".join(buf + [word])
        overflow = (max_words is not None and len(buf) >= max_words) or (
            max_chars is not None and len(candidate) > max_chars
        )
        if overflow and buf:
            # Try to cut after the last weak punctuation inside the buffer.
            cut_at = -1
            for i in range(len(buf) - 1, -1, -1):
                if buf[i].rstrip().endswith(tuple(_SECONDARY_BREAKS)):
                    cut_at = i
                    break
            if cut_at >= 0:
                chunks.append(" ".join(buf[: cut_at + 1]))
                buf = buf[cut_at + 1 :]
            else:
                chunks.append(" ".join(buf))
                buf = []
        buf.append(word)
    if buf:
        chunks.append(" ".join(buf))
    return [chunk for chunk in chunks if chunk]


def _chunk_timings(
    chunks: list[str],
    *,
    start: float,
    end: float,
    words: list | None,
) -> list[tuple[float, float]]:
    """
    Assign start/end to each text chunk.

    Prefer consuming word-level timestamps in order; fall back to proportional
    word-count shares across the parent segment span.
    """
    if not chunks:
        return []
    stream = _word_time_stream(words) if words else []
    usable = [(tok, t0, t1) for tok, t0, t1 in stream if t0 is not None and t1 is not None]
    if usable:
        times: list[tuple[float, float]] = []
        cursor = 0
        for i, chunk in enumerate(chunks):
            n = max(len(_alnum_tokens(chunk)), 1)
            if cursor >= len(usable):
                t = times[-1][1] if times else end
                times.append((t, t if i < len(chunks) - 1 else end))
                continue
            take = min(n, len(usable) - cursor)
            if i == len(chunks) - 1:
                take = len(usable) - cursor
            piece = usable[cursor : cursor + take]
            cursor += take
            t0 = max(start, min(float(piece[0][1]), end))
            t1 = max(t0, min(float(piece[-1][2]), end))
            if times and t0 < times[-1][1]:
                t0 = times[-1][1]
            if t1 < t0:
                t1 = t0
            times.append((t0, t1))
        return times

    weights = [max(len(_alnum_tokens(chunk)), 1) for chunk in chunks]
    total_weight = float(sum(weights))
    cursor = start
    span = end - start
    times = []
    for i, weight in enumerate(weights):
        share = span * (weight / total_weight) if total_weight > 0 else 0.0
        chunk_start = cursor
        chunk_end = cursor + share if i < len(chunks) - 1 else end
        cursor = chunk_end
        if chunk_end < chunk_start:
            chunk_end = chunk_start
        times.append((chunk_start, chunk_end))
    return times


def _split_transcription_segments(
    segments: list[dict],
    *,
    language: str | None,
    max_words: int | None = None,
    max_chars: int | None = None,
    max_duration: float | None = None,
) -> list[dict]:
    """
    Split ASR segments on sentence boundaries for practice-friendly cues.

    Default strategy is punctuation-first (strong breaks only). Optional
    ``max_words`` / ``max_chars`` / ``max_duration`` enable length or duration
    fallbacks for display-oriented subtitle lines.

    Short fragments stay as their own cues (no merge) so dialogue turns are
    not glued onto the previous speaker. Timings prefer per-word timestamps
    when present.
    """
    split_segments: list[dict] = []
    for segment in segments:
        start_raw = segment.get("start")
        end_raw = segment.get("end")
        if start_raw is None or end_raw is None:
            continue
        start = float(start_raw)
        end = float(end_raw)
        if end < start:
            end = start
        text = str(segment.get("text") or "").strip()
        if not text:
            continue

        units = _split_sentence_like_units(text, language)
        chunks: list[str] = []
        length_limits = max_words is not None or max_chars is not None
        for unit in units:
            if length_limits:
                chunks.extend(_split_by_length(unit, max_words=max_words, max_chars=max_chars))
            else:
                chunks.append(unit)
        chunks = [c for c in chunks if c]
        if not chunks:
            chunks = [text]

        # Optional duration guard: further split a single long span by word count.
        total_span = max(0.0, end - start)
        if (
            max_duration is not None
            and max_duration > 0
            and total_span > max_duration
            and len(chunks) == 1
        ):
            words = chunks[0].split()
            if words:
                est_parts = max(2, int(round(total_span / max_duration)))
                size = max(1, len(words) // est_parts)
                expanded: list[str] = []
                for i in range(0, len(words), size):
                    expanded.append(" ".join(words[i : i + size]))
                chunks = expanded or chunks

        word_list = segment.get("words")
        words = word_list if isinstance(word_list, list) else None
        for chunk, (chunk_start, chunk_end) in zip(
            chunks,
            _chunk_timings(chunks, start=start, end=end, words=words),
            strict=True,
        ):
            split_segments.append({"text": chunk, "start": chunk_start, "end": chunk_end})
    return split_segments


def _detect_language(audio, device: str, compute_type: str) -> str:
    import whisperx

    model = whisperx.load_model("tiny", device, compute_type=compute_type)
    # WhisperX models expose language detection via a short transcribe
    result = model.transcribe(audio, language=None, batch_size=8)
    language = result.get("language")
    if not language:
        raise RuntimeError("Failed to detect language from audio")
    return str(language)


def _alnum_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _timed_asr_tokens(
    segments: list[dict],
) -> list[tuple[str, float, float]]:
    """
    Flatten transcription segments into (token, start, end).

    Prefer per-word times when present (from a prior WhisperX align pass).
    Otherwise interpolate evenly across each segment — inaccurate for long
    multi-sentence ASR blobs that include pauses.
    """
    stream: list[tuple[str, float, float]] = []
    for segment in segments:
        text = str(segment.get("text") or "")
        tokens = _alnum_tokens(text)
        if not tokens:
            continue
        start_raw = segment.get("start")
        end_raw = segment.get("end")
        if start_raw is None or end_raw is None:
            continue
        start = float(start_raw)
        end = float(end_raw)
        if end < start:
            end = start
        words = segment.get("words")
        if words:
            for word in words:
                raw = str(word.get("word", ""))
                sub = _alnum_tokens(raw)
                if not sub:
                    continue
                w_start = word.get("start")
                w_end = word.get("end")
                if w_start is None or w_end is None:
                    continue
                t0, t1 = float(w_start), float(w_end)
                if t1 < t0:
                    t1 = t0
                for token in sub:
                    stream.append((token, t0, t1))
            continue
        duration = end - start
        n = len(tokens)
        for i, token in enumerate(tokens):
            t0 = start + duration * i / n
            t1 = start + duration * (i + 1) / n
            stream.append((token, t0, t1))
    return stream


def _segments_with_word_times(
    asr_segments: list[dict],
    align_result: dict,
) -> list[dict]:
    """
    Prefer WhisperX-aligned ASR segments that carry per-word timestamps.

    Falls back to the original ASR segments when alignment produced no words
    (even interpolation in ``_timed_asr_tokens`` still applies).
    """
    aligned = align_result.get("segments") or []
    if any(seg.get("words") for seg in aligned):
        return aligned
    word_segments = align_result.get("word_segments") or []
    if not word_segments or not asr_segments:
        return asr_segments
    # Flat word list only: attach as one synthetic segment spanning ASR bounds.
    starts = [float(w["start"]) for w in word_segments if w.get("start") is not None]
    ends = [float(w["end"]) for w in word_segments if w.get("end") is not None]
    if not starts or not ends:
        return asr_segments
    text = " ".join(str(w.get("word", "")) for w in word_segments).strip()
    return [
        {
            "text": text or str(asr_segments[0].get("text") or ""),
            "start": min(starts),
            "end": max(ends),
            "words": word_segments,
        }
    ]


def _cover_cue(
    starts: list[float | None],
    ends: list[float | None],
    cue_i: int,
    t0: float,
    t1: float,
) -> None:
    prev_start = starts[cue_i]
    prev_end = ends[cue_i]
    starts[cue_i] = t0 if prev_start is None else min(prev_start, t0)
    ends[cue_i] = t1 if prev_end is None else max(prev_end, t1)


def _assign_windows_from_asr(
    cues: list[Cue],
    asr_tokens: list[tuple[str, float, float]],
) -> list[tuple[float | None, float | None]]:
    """
    Map ASR timed tokens onto cues via SequenceMatcher.

    Equal spans match 1:1. Replace spans share the ASR time range across the
    involved cues (handles Alright vs All right). Unmatched cues stay None.
    """
    if not cues:
        return []
    cue_toks: list[tuple[int, str]] = []
    for i, cue in enumerate(cues):
        for token in _alnum_tokens(cue.text):
            cue_toks.append((i, token))

    starts: list[float | None] = [None] * len(cues)
    ends: list[float | None] = [None] * len(cues)
    if not cue_toks or not asr_tokens:
        return list(zip(starts, ends, strict=True))

    cue_only = [t for _, t in cue_toks]
    asr_only = [t for t, _, _ in asr_tokens]
    matcher = SequenceMatcher(a=cue_only, b=asr_only, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                cue_i, _ = cue_toks[i1 + offset]
                _, t0, t1 = asr_tokens[j1 + offset]
                _cover_cue(starts, ends, cue_i, t0, t1)
            continue
        if tag != "replace" or j1 >= j2 or i1 >= i2:
            continue
        span_start = min(t0 for _, t0, _ in asr_tokens[j1:j2])
        span_end = max(t1 for _, _, t1 in asr_tokens[j1:j2])
        # Distribute replace-span time across cues by their token counts.
        counts: dict[int, int] = {}
        order: list[int] = []
        for idx in range(i1, i2):
            cue_i, _ = cue_toks[idx]
            if cue_i not in counts:
                order.append(cue_i)
                counts[cue_i] = 0
            counts[cue_i] += 1
        total = float(sum(counts.values()))
        if total <= 0:
            continue
        cursor = span_start
        span = span_end - span_start
        for cue_i in order:
            share = span * (counts[cue_i] / total)
            t0 = cursor
            t1 = cursor + share
            cursor = t1
            _cover_cue(starts, ends, cue_i, t0, t1)

    return list(zip(starts, ends, strict=True))


def _fill_missing_windows(
    windows: list[tuple[float | None, float | None]],
    *,
    audio_duration: float | None,
) -> list[tuple[float, float]]:
    """Interpolate None windows from neighbors; fall back to full audio."""
    n = len(windows)
    if n == 0:
        return []
    duration = audio_duration if audio_duration is not None and audio_duration > 0 else None
    filled: list[tuple[float | None, float | None]] = list(windows)

    # Forward fill starts from previous end; backward fill ends from next start.
    last_end: float | None = 0.0 if duration is not None else None
    for i in range(n):
        start, end = filled[i]
        if start is None and last_end is not None:
            start = last_end
        if end is None and start is not None:
            # Peek ahead for a known start.
            next_start = None
            for j in range(i + 1, n):
                if filled[j][0] is not None:
                    next_start = filled[j][0]
                    break
            if next_start is not None:
                end = next_start
            elif duration is not None:
                end = duration
            else:
                end = start + _MIN_WINDOW
        if start is not None and end is not None and end < start:
            end = start + _MIN_WINDOW
        filled[i] = (start, end)
        if end is not None:
            last_end = end

    # Backward pass for still-missing starts.
    next_start: float | None = duration
    for i in range(n - 1, -1, -1):
        start, end = filled[i]
        if end is None and next_start is not None:
            end = next_start
        if start is None and end is not None:
            prev_end = None
            for j in range(i - 1, -1, -1):
                if filled[j][1] is not None:
                    prev_end = filled[j][1]
                    break
            start = prev_end if prev_end is not None else max(0.0, end - _MIN_WINDOW)
        if start is not None and end is not None and end < start:
            start = max(0.0, end - _MIN_WINDOW)
        filled[i] = (start, end)
        if start is not None:
            next_start = start

    result: list[tuple[float, float]] = []
    for i, (start, end) in enumerate(filled):
        if start is None or end is None:
            # Last resort: proportional slice of audio / unit interval.
            total = duration if duration is not None else float(max(n, 1))
            start = total * i / n
            end = total * (i + 1) / n
        if end < start + _MIN_WINDOW:
            end = start + _MIN_WINDOW
            if duration is not None:
                end = min(end, duration)
                if end < start:
                    start = end
        result.append((start, end))
    return result


def _enforce_min_windows(
    windows: list[tuple[float, float]],
    *,
    min_duration: float,
    audio_duration: float | None,
) -> list[tuple[float, float]]:
    """Expand windows shorter than ``min_duration`` into neighboring gaps."""
    if not windows or min_duration <= 0:
        return windows
    n = len(windows)
    out = list(windows)
    limit = audio_duration if audio_duration is not None and audio_duration > 0 else None

    for i in range(n):
        start, end = out[i]
        if end - start >= min_duration:
            continue
        need = min_duration - (end - start)
        next_start = out[i + 1][0] if i + 1 < n else (limit if limit is not None else end + need)
        room_right = max(0.0, next_start - end)
        take = min(need, room_right)
        end += take
        need -= take
        if need > 0:
            prev_end = out[i - 1][1] if i > 0 else 0.0
            room_left = max(0.0, start - prev_end)
            take = min(need, room_left)
            start -= take
            need -= take
        if need > 0:
            end += need
            if limit is not None:
                end = min(end, limit)
                if end < start + min_duration:
                    start = max(0.0, end - min_duration)
        if end < start:
            end = start
        out[i] = (start, end)
    return out


def _clip_speech_spans(
    spans: list[tuple[float, float]],
    lo: float,
    hi: float,
) -> list[tuple[float, float]]:
    """Return speech spans clipped to ``[lo, hi]``, dropping empty leftovers."""
    if hi <= lo or not spans:
        return []
    clipped: list[tuple[float, float]] = []
    for start, end in spans:
        a = max(start, lo)
        b = min(end, hi)
        if b - a >= _MIN_WINDOW:
            clipped.append((a, b))
    return clipped


def _proportion_on_speech(
    cues: list[Cue],
    speech_spans: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Map cues onto speech spans by text-length weight (same idea as realign)."""
    segments = assign_windows(
        cues,
        mode="realign",
        margin=0.0,
        speech_spans=speech_spans,
    )
    return [(float(s["start"]), float(s["end"])) for s in segments]


def _gap_speech_islands(
    speech_spans: list[tuple[float, float]],
    gap_start: float,
    gap_end: float,
    *,
    edge_guard: float = _GAP_EDGE_GUARD,
) -> list[tuple[float, float]]:
    """
    Speech inside a fill gap, preferring islands that are not just the next cue.

    Islands that only begin within ``edge_guard`` of ``gap_end`` are dropped so
    a missing middle line does not search on the following utterance.
    """
    islands = _clip_speech_spans(speech_spans, gap_start, gap_end)
    if not islands:
        return []
    interior = [sp for sp in islands if sp[0] < gap_end - edge_guard]
    return interior or islands


def _windows_for_unmatched_run(
    cues: list[Cue],
    gap_start: float,
    gap_end: float,
    speech_spans: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """Place unmatched cues on VAD speech inside their fill gap."""
    islands = _gap_speech_islands(speech_spans, gap_start, gap_end)
    if not islands:
        return None
    if len(cues) == 1:
        # One cue + several islands: pick the longest interior island so the
        # phrase is not stretched across silence into the next line.
        best = max(islands, key=lambda sp: sp[1] - sp[0])
        return [best]
    return _proportion_on_speech(cues, islands)


def _refine_windows_with_speech(
    rough: list[tuple[float | None, float | None]],
    filled: list[tuple[float, float]],
    cues: list[Cue],
    speech_spans: list[tuple[float, float]],
    *,
    max_search_window: float = _MAX_SEARCH_WINDOW,
) -> list[tuple[float, float]]:
    """
    Narrow ASR fill / oversized windows onto VAD speech.

    Unmatched runs (rough times were None) are remapped onto speech islands
    inside their neighbor gap. Matched windows longer than ``max_search_window``
    are clipped to overlapping speech when available.
    """
    if not filled or not speech_spans:
        return filled
    n = len(filled)
    out = list(filled)
    unmatched = [start is None and end is None for start, end in rough]

    i = 0
    while i < n:
        if not unmatched[i]:
            start, end = out[i]
            if end - start > max_search_window:
                islands = _clip_speech_spans(speech_spans, start, end)
                if islands:
                    # Keep chronological coverage but drop long silence holes.
                    out[i] = (islands[0][0], islands[-1][1])
            i += 1
            continue

        j = i
        while j < n and unmatched[j]:
            j += 1
        gap_start = out[i][0]
        gap_end = out[j - 1][1]
        remapped = _windows_for_unmatched_run(cues[i:j], gap_start, gap_end, speech_spans)
        if remapped is not None and len(remapped) == j - i:
            for offset, window in enumerate(remapped):
                out[i + offset] = window
        i = j
    return out


def build_script_align_segments(
    cues: list[Cue],
    transcription_segments: list[dict],
    *,
    margin: float = 0.5,
    audio_duration: float | None = None,
    min_search_duration: float = _MIN_SEARCH_DURATION,
    speech_spans: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """
    Build WhisperX segments from script cues + ASR-derived search windows.

    Transcription is used only for timing anchors. Segment text is always the
    original cue text so forced alignment follows the subtitle, not ASR wording.
    When ``speech_spans`` is provided, unmatched / oversized fill windows are
    narrowed onto VAD speech so short cues are not force-aligned at the next
    utterance after a long silence.
    """
    asr_tokens = _timed_asr_tokens(transcription_segments)
    rough = _assign_windows_from_asr(cues, asr_tokens)
    windows = _fill_missing_windows(rough, audio_duration=audio_duration)
    if speech_spans:
        windows = _refine_windows_with_speech(rough, windows, cues, speech_spans)
    windows = _enforce_min_windows(
        windows,
        min_duration=min_search_duration,
        audio_duration=audio_duration,
    )

    segments: list[dict] = []
    for cue, (start, end) in zip(cues, windows, strict=True):
        win_start = max(0.0, start - margin)
        win_end = end + margin
        if audio_duration is not None:
            win_end = min(win_end, audio_duration)
        if win_end < win_start + _MIN_WINDOW:
            win_end = win_start + _MIN_WINDOW
            if audio_duration is not None:
                win_end = min(win_end, audio_duration)
                if win_end < win_start:
                    win_start = win_end
        segments.append({"text": cue.text, "start": win_start, "end": win_end})
    return segments


def _group_short_cues(
    cues: list[Cue],
    *,
    max_tokens: int = _SHORT_CUE_TOKENS,
) -> list[list[int]]:
    """
    Group cue indices for temporary merge.

    Short cues attach to the previous group when possible; leading shorts absorb
    the following cue so they are never aligned alone.
    """
    n = len(cues)
    if n == 0:
        return []
    short = [len(_alnum_tokens(c.text)) <= max_tokens for c in cues]
    groups: list[list[int]] = []
    i = 0
    while i < n:
        if not short[i]:
            groups.append([i])
            i += 1
            continue
        if groups:
            groups[-1].append(i)
            i += 1
            continue
        group = [i]
        i += 1
        while i < n and short[i]:
            group.append(i)
            i += 1
        if i < n:
            group.append(i)
            i += 1
        groups.append(group)
    return groups


def _merge_align_segments(
    cues: list[Cue],
    segments: list[dict],
    *,
    max_tokens: int = _SHORT_CUE_TOKENS,
) -> tuple[list[dict], list[list[int]]]:
    """Merge short-cue segments for WhisperX; return (merged, groups)."""
    groups = _group_short_cues(cues, max_tokens=max_tokens)
    if all(len(g) == 1 for g in groups):
        return segments, groups

    merged: list[dict] = []
    for group in groups:
        parts = [segments[i] for i in group]
        text = " ".join(cues[i].text for i in group)
        start = min(float(p["start"]) for p in parts)
        end = max(float(p["end"]) for p in parts)
        if end < start + _MIN_WINDOW:
            end = start + _MIN_WINDOW
        merged.append({"text": text, "start": start, "end": end})
    return merged, groups


def _collect_words(
    aligned_segments: list[dict],
    word_segments: list[dict] | None,
) -> list[dict]:
    if word_segments:
        return word_segments
    words: list[dict] = []
    for segment in aligned_segments:
        words.extend(segment.get("words") or [])
    return words


def _word_time_stream(words: list[dict]) -> list[tuple[str, float | None, float | None]]:
    """Expand aligner words into (token, start, end); share times across subtokens."""
    stream: list[tuple[str, float | None, float | None]] = []
    for word in words:
        raw = str(word.get("word", ""))
        tokens = _alnum_tokens(raw)
        if not tokens:
            continue
        start_raw = word.get("start")
        end_raw = word.get("end")
        start = float(start_raw) if start_raw is not None else None
        end = float(end_raw) if end_raw is not None else None
        for token in tokens:
            stream.append((token, start, end))
    return stream


def _apply_from_words(
    cues: list[Cue],
    words: list[dict],
) -> tuple[list[Cue], list[bool]] | None:
    """
    Map cues onto aligner words in order.

    Returns (cues, exact_flags) or None if words are unusable. ``exact_flags[i]``
    is False when cue i fell back to count-based consumption after a token mismatch.
    """
    stream = _word_time_stream(words)
    if not stream:
        return None

    updated: list[Cue] = []
    exact_flags: list[bool] = []
    idx = 0
    for cue in cues:
        need = _alnum_tokens(cue.text)
        if not need:
            updated.append(cue)
            exact_flags.append(True)
            continue
        if idx >= len(stream):
            updated.append(cue)
            exact_flags.append(False)
            continue

        start_idx = idx
        matched = 0
        starts: list[float] = []
        ends: list[float] = []
        while idx < len(stream) and matched < len(need):
            token, start, end = stream[idx]
            if token != need[matched]:
                break
            if start is not None:
                starts.append(start)
            if end is not None:
                ends.append(end)
            matched += 1
            idx += 1

        exact = matched == len(need)
        if not exact:
            # Token text drifted; still consume by count to keep cue boundaries.
            idx = start_idx
            starts = []
            ends = []
            for _ in need:
                if idx >= len(stream):
                    break
                _, start, end = stream[idx]
                if start is not None:
                    starts.append(start)
                if end is not None:
                    ends.append(end)
                idx += 1

        if not starts or not ends:
            updated.append(cue)
            exact_flags.append(False)
            continue
        start = min(starts)
        end = max(ends)
        if end < start:
            end = start
        updated.append(Cue(index=cue.index, text=cue.text, start=start, end=end))
        exact_flags.append(exact)
    return updated, exact_flags


def _apply_one_to_one(cues: list[Cue], aligned_segments: list[dict]) -> list[Cue]:
    updated: list[Cue] = []
    count = min(len(cues), len(aligned_segments))
    for i in range(count):
        segment = aligned_segments[i]
        start = float(segment.get("start", cues[i].start))
        end = float(segment.get("end", cues[i].end))
        if end < start:
            end = start
        updated.append(Cue(index=cues[i].index, text=cues[i].text, start=start, end=end))
    for cue in cues[count:]:
        updated.append(cue)
    return updated


def _apply_groups_proportional(
    cues: list[Cue],
    aligned_segments: list[dict],
    groups: list[list[int]],
) -> list[Cue]:
    """Split each merged segment's time across its cues by token weight."""
    by_index: dict[int, Cue] = {c.index: c for c in cues}
    times: dict[int, tuple[float, float]] = {}
    for seg, group in zip(aligned_segments, groups, strict=False):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        if seg_end < seg_start:
            seg_end = seg_start
        weights = [max(len(_alnum_tokens(cues[i].text)), 1) for i in group]
        total = float(sum(weights))
        cursor = seg_start
        span = seg_end - seg_start
        for j, cue_i in enumerate(group):
            share = span * (weights[j] / total) if total else span / max(len(group), 1)
            t0 = cursor
            t1 = cursor + share if j < len(group) - 1 else seg_end
            cursor = t1
            times[cue_i] = (t0, t1)

    updated: list[Cue] = []
    for i, cue in enumerate(cues):
        if i in times:
            start, end = times[i]
            updated.append(Cue(index=cue.index, text=cue.text, start=start, end=end))
        else:
            updated.append(by_index.get(cue.index, cue))
    return updated


def _cue_window_fallback(
    cues: list[Cue],
    search_segments: list[dict],
) -> list[tuple[float, float]]:
    """Per-cue fallback times from unmerged search windows."""
    out: list[tuple[float, float]] = []
    for i, cue in enumerate(cues):
        if i < len(search_segments):
            start = float(search_segments[i].get("start", cue.start))
            end = float(search_segments[i].get("end", cue.end))
        else:
            start, end = cue.start, cue.end
        if end < start:
            end = start
        out.append((start, end))
    return out


def _repair_thin_cues(
    aligned: list[Cue],
    *,
    exact_flags: list[bool],
    fallback_windows: list[tuple[float, float]],
) -> list[Cue]:
    """
    Replace failed / tiny alignments with search-window fallbacks.

    A cue is repaired when its duration is below ``_MIN_ALIGNED_DURATION``, or
    when token matching fell back to count-based consume and the span is still
    shorter than the search window floor.
    """
    updated: list[Cue] = []
    for i, cue in enumerate(aligned):
        dur = cue.end - cue.start
        exact = exact_flags[i] if i < len(exact_flags) else True
        fb_start, fb_end = fallback_windows[i]
        if fb_end < fb_start:
            fb_end = fb_start
        thin = dur < _MIN_ALIGNED_DURATION
        weak = (not exact) and dur < _MIN_SEARCH_DURATION
        if thin or weak:
            updated.append(Cue(index=cue.index, text=cue.text, start=fb_start, end=fb_end))
        else:
            updated.append(cue)
    return updated


def _apply_aligned_times(
    cues: list[Cue],
    aligned_segments: list[dict],
    word_segments: list[dict] | None = None,
    *,
    search_segments: list[dict] | None = None,
    groups: list[list[int]] | None = None,
) -> list[Cue]:
    """
    Apply WhisperX alignment times back onto original cues.

    WhisperX splits multi-sentence cues into separate segments, so index-aligned
    mapping is wrong. Prefer word-level remapping; fall back to 1:1 (or group
    proportional split) when words are unavailable (e.g. mocks).
    """
    fallback_src = search_segments if search_segments is not None else []
    if not fallback_src and groups is None:
        fallback_src = aligned_segments
    fallback_windows = (
        _cue_window_fallback(cues, fallback_src)
        if fallback_src
        else [(c.start, c.end) for c in cues]
    )

    words = _collect_words(aligned_segments, word_segments)
    if words:
        remapped = _apply_from_words(cues, words)
        if remapped is not None:
            updated, exact_flags = remapped
            return _repair_thin_cues(
                updated,
                exact_flags=exact_flags,
                fallback_windows=fallback_windows,
            )

    if groups is not None and any(len(g) > 1 for g in groups):
        return _apply_groups_proportional(cues, aligned_segments, groups)
    return _apply_one_to_one(cues, aligned_segments)


def _trim_overlaps(cues: list[Cue]) -> list[Cue]:
    """
    Enforce a non-overlapping, start-monotonic timeline.

    Normal overlap (start_N <= start_{N+1} < end_N): shorten end_N to
    start_{N+1}. Inversion (start_{N+1} < start_N): advance start_{N+1} to
    end_N (after ensuring end_N >= start_N). Never leaves a later cue starting
    before an earlier cue.
    """
    if len(cues) < 2:
        return cues

    updated = [
        Cue(
            index=c.index,
            text=c.text,
            start=c.start,
            end=c.end if c.end >= c.start else c.start,
        )
        for c in cues
    ]

    for i in range(len(updated) - 1):
        cur = updated[i]
        nxt = updated[i + 1]
        if cur.end <= nxt.start:
            continue
        if cur.start <= nxt.start:
            new_end = nxt.start
            if new_end < cur.start:
                new_end = cur.start
            if new_end != cur.end:
                updated[i] = Cue(index=cur.index, text=cur.text, start=cur.start, end=new_end)
            continue
        # Inversion: later cue starts before earlier cue.
        new_start = cur.end if cur.end >= cur.start else cur.start
        new_end = nxt.end if nxt.end >= new_start else new_start
        updated[i + 1] = Cue(index=nxt.index, text=nxt.text, start=new_start, end=new_end)
        # Earlier cue may still extend past the new boundary.
        if updated[i].end > new_start:
            updated[i] = Cue(
                index=cur.index,
                text=cur.text,
                start=cur.start,
                end=max(cur.start, new_start),
            )
    return updated


def _fill_gaps(
    cues: list[Cue],
    *,
    audio_duration: float | None = None,
) -> list[Cue]:
    """
    Extend each cue end to the next cue start so the timeline is contiguous.

    Starts are never moved. The last cue ends at ``audio_duration`` when that
    value is known and greater than the cue start; otherwise its end is kept
    (clamped to start if needed).
    """
    if not cues:
        return cues
    updated: list[Cue] = []
    last_i = len(cues) - 1
    for i, cue in enumerate(cues):
        if i < last_i:
            end = cues[i + 1].start
        elif audio_duration is not None and audio_duration > cue.start:
            end = audio_duration
        else:
            end = cue.end
        if end < cue.start:
            end = cue.start
        if end == cue.end:
            updated.append(cue)
        else:
            updated.append(Cue(index=cue.index, text=cue.text, start=cue.start, end=end))
    return updated


def _shift_cues(cues: list[Cue], offset: float) -> list[Cue]:
    if offset == 0:
        return cues
    return [
        Cue(
            index=cue.index,
            text=cue.text,
            start=cue.start + offset,
            end=cue.end + offset,
        )
        for cue in cues
    ]


def estimate_global_offset(
    cues: list[Cue],
    asr_segments: list[dict],
    *,
    min_matches: int = _MIN_OFFSET_MATCHES,
    ignore_below: float = _MIN_ABS_OFFSET,
) -> float:
    """
    Estimate a constant timeline shift so cue times land near ASR speech.

    Matches cue text to ASR via the same token aligner used for .txt windows,
    then returns the median of (asr_start - cue.start). Returns 0 when too few
    cues match or the median drift is below ``ignore_below``.
    """
    if not cues or not asr_segments:
        return 0.0
    windows = _assign_windows_from_asr(cues, _timed_asr_tokens(asr_segments))
    deltas: list[float] = []
    for cue, (start, _) in zip(cues, windows, strict=True):
        if start is None:
            continue
        deltas.append(float(start) - cue.start)
    if len(deltas) < min_matches:
        return 0.0
    offset = float(median(deltas))
    if abs(offset) < ignore_below:
        return 0.0
    return offset


def align_file(
    media: str | Path,
    subtitle: str | Path | None = None,
    output: str | Path | None = None,
    *,
    language: str | None = None,
    detect_language: bool = False,
    device: str = "auto",
    compute_type: str | None = None,
    model_name: str = "small",
    margin: float = 0.5,
    fill_gaps: bool = False,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    offset: float | None = None,
    auto_offset: bool = True,
    print_progress: bool = False,
    max_words: int | None = None,
    max_chars: int | None = None,
    max_duration: float | None = None,
) -> Path:
    """
    Align subtitle timestamps to media, or transcribe media when no subtitle is given.

    When ``fill_gaps`` is True, each cue end is extended to the next cue start
    and the last cue ends at the kept media region (after trim).

    ``trim_start`` / ``trim_end`` drop leading/trailing seconds before alignment
    (e.g. podcast intros not present in the script). Output times are remapped
    onto the original media timeline.

    When ``subtitle`` is omitted, this runs WhisperX transcription and writes
    timed subtitles directly. Audio-only cues default to strong-punctuation
    sentence splits; pass ``max_words`` / ``max_chars`` / ``max_duration`` to
    also cap line length or duration (display-oriented subtitles).

    For subtitle inputs, ``.srt`` / ``.lrc`` apply a constant timeline shift
    before refine: use ``offset`` to set it manually, or leave
    ``auto_offset=True`` (default) to estimate from a short Whisper
    transcription. ``.txt`` inputs ignore these options (windows come from
    transcription already).

    Returns the output path written.
    """
    media_path = Path(media)
    subtitle_path = Path(subtitle) if subtitle is not None else None
    if not media_path.is_file():
        raise FileNotFoundError(f"Media not found: {media_path}")
    if subtitle_path is not None and not subtitle_path.is_file():
        raise FileNotFoundError(f"Subtitle not found: {subtitle_path}")

    if language is None and not detect_language:
        raise ValueError("Provide --language or set detect_language=True")

    subtitle_suffix = subtitle_path.suffix.lower() if subtitle_path is not None else ""
    use_transcription_windows = subtitle_suffix == ".txt"

    out_path = Path(output) if output else _default_output(media_path, subtitle_path)
    if out_path.suffix.lower() not in {".srt", ".lrc"}:
        out_path = out_path.with_suffix(".srt")

    resolved = resolve_device(device)
    ctype = default_compute_type(resolved, compute_type)

    full_audio = load_audio(media_path)
    full_duration = audio_duration(full_audio)
    audio = trim_audio(full_audio, trim_start=trim_start, trim_end=trim_end)
    duration = audio_duration(audio)
    cue_time_offset = trim_start

    try:
        import whisperx
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "whisperx is required for alignment. Install with: "
            "pip install 'sub-align[align]' (or [cpu]/[gpu])"
        ) from exc

    lang = language or _detect_language(audio, resolved, ctype)

    if subtitle_path is None:
        model = whisperx.load_model(model_name, resolved, compute_type=ctype, language=lang)
        transcription = model.transcribe(audio, language=lang, batch_size=16)
        asr_segments = transcription.get("segments") or []
        # Word-align ASR so split cues use real word boundaries instead of
        # proportional timing within each Whisper segment.
        if asr_segments:
            align_model, metadata = whisperx.load_align_model(
                language_code=lang,
                device=resolved,
            )
            asr_aligned = whisperx.align(
                asr_segments,
                align_model,
                metadata,
                audio,
                resolved,
                return_char_alignments=False,
                print_progress=print_progress,
            )
            asr_segments = _segments_with_word_times(asr_segments, asr_aligned)
        split_segments = _split_transcription_segments(
            asr_segments,
            language=lang,
            max_words=max_words,
            max_chars=max_chars,
            max_duration=max_duration,
        )
        updated = _cues_from_transcription(split_segments)
        if not updated:
            raise ValueError("No timed transcription segments found in media")
        if fill_gaps:
            updated = _fill_gaps(updated, audio_duration=duration)
        updated = _shift_cues(updated, cue_time_offset)
        _write_cues(out_path, updated)
        return out_path

    # Cue times for .srt/.lrc are on the original timeline; shift into the
    # trimmed coordinate system used by WhisperX on ``audio``.
    cues = _read_cues(subtitle_path, audio_duration=full_duration)
    if not cues:
        raise ValueError(f"No cues found in {subtitle_path}")
    if cue_time_offset:
        cues = [
            Cue(
                index=cue.index,
                text=cue.text,
                start=max(0.0, cue.start - cue_time_offset),
                end=max(0.0, cue.end - cue_time_offset),
            )
            for cue in cues
        ]

    align_model, metadata = whisperx.load_align_model(
        language_code=lang,
        device=resolved,
    )

    if use_transcription_windows:
        # Transcribe for rough windows, then word-align the ASR text so long
        # multi-sentence segments are not evenly interpolated across pauses.
        # Script text is still what we force-align onto the audio next.
        speech_spans = detect_speech_spans(audio)
        model = whisperx.load_model(model_name, resolved, compute_type=ctype, language=lang)
        transcription = model.transcribe(audio, language=lang, batch_size=16)
        asr_segments = transcription.get("segments") or []
        if asr_segments:
            asr_aligned = whisperx.align(
                asr_segments,
                align_model,
                metadata,
                audio,
                resolved,
                return_char_alignments=False,
                print_progress=print_progress,
            )
            asr_segments = _segments_with_word_times(asr_segments, asr_aligned)
        segments = build_script_align_segments(
            cues,
            asr_segments,
            margin=margin,
            audio_duration=duration,
            speech_spans=speech_spans,
        )
    else:
        # Constant shift first (manual or ASR-estimated), then refine locally.
        applied_offset = 0.0
        offset_source = "none"
        if offset is not None:
            applied_offset = float(offset)
            offset_source = "manual"
        elif auto_offset:
            model = whisperx.load_model(model_name, resolved, compute_type=ctype, language=lang)
            transcription = model.transcribe(audio, language=lang, batch_size=16)
            asr_segments = transcription.get("segments") or []
            applied_offset = estimate_global_offset(cues, asr_segments)
            offset_source = "auto"
        if applied_offset:
            cues = _shift_cues(cues, applied_offset)
            print(
                f"sub-align: applied global offset {applied_offset:+.3f}s ({offset_source})",
                file=sys.stderr,
            )
        segments = assign_windows(
            cues,
            mode="refine",
            margin=margin,
            audio_duration=duration,
        )

    search_segments = segments
    groups: list[list[int]] | None = None
    if use_transcription_windows:
        # Short script lines get a wider align span by merging with neighbors;
        # word remapping still assigns times back onto the original cues.
        segments, groups = _merge_align_segments(cues, segments)

    result = whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        resolved,
        return_char_alignments=False,
        print_progress=print_progress,
    )
    aligned = result.get("segments") or []
    word_segments = result.get("word_segments") or None
    updated = _trim_overlaps(
        _apply_aligned_times(
            cues,
            aligned,
            word_segments=word_segments,
            search_segments=search_segments,
            groups=groups,
        )
    )
    if fill_gaps:
        updated = _fill_gaps(updated, audio_duration=duration)
    updated = _shift_cues(updated, cue_time_offset)
    _write_cues(out_path, updated)
    return out_path


def _default_output(media_path: Path, subtitle_path: Path | None) -> Path:
    if subtitle_path is None:
        return media_path.with_name(f"{media_path.stem}.asr.srt")
    return subtitle_path.with_name(f"{subtitle_path.stem}.aligned{subtitle_path.suffix}")
