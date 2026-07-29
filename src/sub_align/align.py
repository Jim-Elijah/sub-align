from __future__ import annotations

import re
from pathlib import Path

from sub_align.audio import audio_duration, load_audio
from sub_align.device import default_compute_type, resolve_device
from sub_align.formats import lrc, srt
from sub_align.models import Cue
from sub_align.vad import detect_speech_spans
from sub_align.windows import assign_windows

# WhisperX may split one input cue into multiple sentence segments; remap via
# alphanumeric tokens so contractions like "I'm" stay consistent across sides.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


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


def _apply_from_words(cues: list[Cue], words: list[dict]) -> list[Cue] | None:
    """Map cues onto aligner words in order. Returns None if words are unusable."""
    stream = _word_time_stream(words)
    if not stream:
        return None

    updated: list[Cue] = []
    idx = 0
    for cue in cues:
        need = _alnum_tokens(cue.text)
        if not need:
            updated.append(cue)
            continue
        if idx >= len(stream):
            updated.append(cue)
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

        if matched != len(need):
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
            continue
        start = min(starts)
        end = max(ends)
        if end < start:
            end = start
        updated.append(Cue(index=cue.index, text=cue.text, start=start, end=end))
    return updated


def _apply_one_to_one(cues: list[Cue], aligned_segments: list[dict]) -> list[Cue]:
    updated: list[Cue] = []
    count = min(len(cues), len(aligned_segments))
    for i in range(count):
        segment = aligned_segments[i]
        start = float(segment.get("start", cues[i].start))
        end = float(segment.get("end", cues[i].end))
        if end < start:
            end = start
        updated.append(
            Cue(index=cues[i].index, text=cues[i].text, start=start, end=end)
        )
    for cue in cues[count:]:
        updated.append(cue)
    return updated


def _apply_aligned_times(
    cues: list[Cue],
    aligned_segments: list[dict],
    word_segments: list[dict] | None = None,
) -> list[Cue]:
    """
    Apply WhisperX alignment times back onto original cues.

    WhisperX splits multi-sentence cues into separate segments, so index-aligned
    mapping is wrong. Prefer word-level remapping; fall back to 1:1 when words
    are unavailable (e.g. mocks).
    """
    words = _collect_words(aligned_segments, word_segments)
    if words:
        remapped = _apply_from_words(cues, words)
        if remapped is not None:
            return remapped
    return _apply_one_to_one(cues, aligned_segments)


def _trim_overlaps(cues: list[Cue]) -> list[Cue]:
    """
    Resolve adjacent overlaps by shortening the earlier cue only.

    If cue N ends after cue N+1 starts, set end_N = start_{N+1}. Never moves
    starts; if that would make end < start, clamp end to start.
    """
    if len(cues) < 2:
        return cues
    updated: list[Cue] = []
    for i, cue in enumerate(cues):
        end = cue.end
        if i + 1 < len(cues) and end > cues[i + 1].start:
            end = cues[i + 1].start
        if end < cue.start:
            end = cue.start
        if end == cue.end:
            updated.append(cue)
        else:
            updated.append(Cue(index=cue.index, text=cue.text, start=cue.start, end=end))
    return updated


def align_file(
    media: str | Path,
    subtitle: str | Path,
    output: str | Path | None = None,
    *,
    language: str | None = None,
    detect_language: bool = False,
    device: str = "auto",
    compute_type: str | None = None,
    mode: str = "realign",
    margin: float = 0.5,
    vad_method: str = "energy",
    print_progress: bool = False,
) -> Path:
    """
    Align subtitle file timestamps to media using WhisperX forced alignment.

    Returns the output path written.
    """
    media_path = Path(media)
    subtitle_path = Path(subtitle)
    if not media_path.is_file():
        raise FileNotFoundError(f"Media not found: {media_path}")
    if not subtitle_path.is_file():
        raise FileNotFoundError(f"Subtitle not found: {subtitle_path}")

    if language is None and not detect_language:
        raise ValueError("Provide --language or set detect_language=True")

    out_path = Path(output) if output else _default_output(subtitle_path)
    if out_path.suffix.lower() not in {".srt", ".lrc"}:
        # Keep original format when output path omits extension
        out_path = out_path.with_suffix(subtitle_path.suffix)

    resolved = resolve_device(device)
    ctype = default_compute_type(resolved, compute_type)

    audio = load_audio(media_path)
    duration = audio_duration(audio)

    cues = _read_cues(subtitle_path, audio_duration=duration)
    if not cues:
        raise ValueError(f"No cues found in {subtitle_path}")

    speech_spans = None
    if mode == "realign":
        speech_spans = detect_speech_spans(audio, method=vad_method)
        if not speech_spans:
            speech_spans = [(0.0, duration)] if duration > 0 else []

    segments = assign_windows(
        cues,
        mode=mode,
        margin=margin,
        speech_spans=speech_spans,
        audio_duration=duration,
    )

    try:
        import whisperx
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "whisperx is required for alignment. Install with: "
            "pip install 'sub-align[align]' (or [cpu]/[gpu])"
        ) from exc

    lang = language or _detect_language(audio, resolved, ctype)
    align_model, metadata = whisperx.load_align_model(
        language_code=lang,
        device=resolved,
    )
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
        _apply_aligned_times(cues, aligned, word_segments=word_segments)
    )
    _write_cues(out_path, updated)
    return out_path


def _default_output(subtitle_path: Path) -> Path:
    return subtitle_path.with_name(f"{subtitle_path.stem}.aligned{subtitle_path.suffix}")
